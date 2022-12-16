import logging
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, Iterable, Set, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.nn.quantized._reference as nnqr
from torch.fx import GraphModule, Node
from torch.quantization import (DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS,
                                QConfig)

from torch_quant.observer import MinMaxObserver, Observer

LOGGER = logging.getLogger(__name__)

QUANTIZABLE_MODULE_TYPES = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.Linear,
)

REF_TO_QUANT_MAP = {
    nnqr.Conv1d: nnq.Conv1d,
    nnqr.Conv2d: nnq.Conv2d,
    nnqr.Conv3d: nnq.Conv3d,
    nnqr.Linear: nnq.Linear,
}


def _locate_parent(root: nn.Module, full_path: str) -> Tuple[nn.Module, str]:
    parent = root
    path = full_path.split('.')
    for p in path[:-1]:
        parent = getattr(parent, p)
    return parent, path[-1]


def _add_module(root: nn.Module, full_path: str, module: nn.Module) -> None:
    parent, name = _locate_parent(root, full_path)
    parent.add_module(name, module)


def _register_buffer(root: nn.Module, full_path: str, tensor: torch.Tensor) -> None:
    parent, name = _locate_parent(root, full_path)
    parent.register_buffer(name, tensor)


class GraphModContext:
    def __init__(self, gm: GraphModule, root: nn.Module) -> None:
        self.gm = gm
        self.root = root
        self.modules = dict(self.root.named_modules())

    def nodes_by_module_type(self, module_types: Iterable[Type[nn.Module]]) -> Iterable[Node]:
        for node in self.gm.graph.nodes:
            if node.op != 'call_module':
                continue
            if any(isinstance(self.modules.get(node.target), t) for t in module_types):
                yield node

    def nodes_by_function_type(self, function_types: Iterable[Callable]) -> Iterable[Node]:
        for node in self.gm.graph.nodes:
            if node.op != 'call_function':
                continue
            if any(node.target == f for f in function_types):
                yield node

    def nodes_by_method_names(self, method_names: Iterable[str]) -> Iterable[Node]:
        for node in self.gm.graph.nodes:
            if node.op == 'call_method':
                continue
            if any(node.target == m for m in method_names):
                yield node

    def get_attr(self, full_path: str) -> Any:
        parent, name = _locate_parent(self.gm, full_path)
        return getattr(parent, name)

    # TODO(litan.ls): make sure not override existing module?
    def add_module(self, full_path: str, module: nn.Module) -> None:
        _add_module(self.gm, full_path, module)
        _add_module(self.root, full_path, module)
        self.modules[full_path] = module

    def replace_module(self, full_path: str, module: nn.Module) -> None:
        _add_module(self.gm, full_path, module)
        self.modules[full_path] = module

    def register_buffer(self, full_path: str, value: torch.Tensor) -> None:
        cloned = value.clone().detach()
        _register_buffer(self.gm, full_path, cloned)
        _register_buffer(self.root, full_path, cloned)

    def get_or_create_module(self, full_path: str, constructor: Callable[[], nn.Module]) -> nn.Module:
        for n, m in self.root.named_modules():
            if n == full_path:
                _add_module(self.gm, full_path, m)
                return m

        m = constructor()
        self.add_module(full_path, m)
        return m


GraphModPass = Callable[[GraphModContext], None]


def modify_graph(gm: GraphModule, root: nn.Module, passes: Iterable[GraphModPass]) -> None:
    ctx = GraphModContext(gm=gm, root=root)
    for p in passes:
        p(ctx)


# Some basic and generic graph modification passes:

# set QConfig to quantizable modules so we can reuse nn.intrinsic/qat modules
# TODO(litan.ls): support other observer type and dtype
def set_qconfig(ctx: GraphModContext) -> None:
    for node in ctx.nodes_by_module_type(QUANTIZABLE_MODULE_TYPES):
        m = ctx.modules.get(node.target)
        m.qconfig = QConfig(activation=None, weight=partial(
            MinMaxObserver, dtype=torch.qint8))


def insert_act_observer(ctx: GraphModContext) -> None:
    # key: activation node to be observed, value: consumers of observed activation
    # note that not all consumers of original activation need consum observed activation.
    act_nodes: Dict[Node, Set[Node]] = defaultdict(set)
    for node in ctx.nodes_by_module_type(QUANTIZABLE_MODULE_TYPES):
        for arg in node.args:
            if isinstance(arg, Node):
                act_nodes[arg].add(node)
        act_nodes[node].update(node.users)
    for act in act_nodes:
        # TODO(litan.ls): act.op == call_method
        if act.op == 'call_function':
            ob_path = f'{act.name}_ob'
        else:
            ob_path = f'{act.target}_ob'

        ob_ctr = partial(MinMaxObserver, dtype=torch.quint8)
        _ = ctx.get_or_create_module(ob_path, ob_ctr)
        with ctx.gm.graph.inserting_after(act):
            ob_node = ctx.gm.graph.call_module(ob_path, (act, ))
        act.replace_all_uses_with(
            ob_node, delete_user_cb=lambda user: user in act_nodes[act])
        LOGGER.debug(f'insert ob({ob_path}) after {act.op}({act.target})')
    # TODO(litan.ls): handle output node
    ctx.gm.recompile()


def observer_to_qdq(ctx: GraphModContext) -> None:
    for node in ctx.nodes_by_module_type([Observer]):
        ob: Observer = ctx.modules[node.target]
        scale_name = f'{node.target}_scale'
        zero_point_name = f'{node.target}_zero_point'
        ctx.register_buffer(scale_name, ob.qparams.scale)
        ctx.register_buffer(zero_point_name, ob.qparams.zero_point)
        with ctx.gm.graph.inserting_before(node):
            scale_node = ctx.gm.graph.get_attr(scale_name)
            zero_point_node = ctx.gm.graph.get_attr(zero_point_name)
            q_inputs = (node.args[0], scale_node, zero_point_node, ob.dtype)
            q = ctx.gm.graph.call_function(torch.quantize_per_tensor, q_inputs)
            dq = ctx.gm.graph.call_method('dequantize', (q,))
            node.replace_all_uses_with(dq)
            ctx.gm.graph.erase_node(node)
        LOGGER.debug(
            f'''ob({node.target}) -> qdp: scale={ob.qparams.scale}, zero_point={ob.qparams.zero_point}''')
    ctx.gm.recompile()


def quantizable_module_to_ref(ctx: GraphModContext) -> None:
    for node in ctx.nodes_by_module_type(QUANTIZABLE_MODULE_TYPES):
        src = ctx.modules[node.target]
        dst_type = DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS.get(
            type(src))
        if dst_type is None:
            raise ValueError(
                f'module type {type(src)} is not supported for static quantization.')
        w_ob = src.qconfig.weight()
        w_ob(src.weight)
        dst = dst_type.from_float(src, w_ob.qparams._asdict())
        # TODO(litan.ls): copy forward hooks
        ctx.replace_module(node.target, dst)
        LOGGER.debug(f'to_ref: {node.target}({type(src)}->{dst_type})')
    ctx.gm.recompile()


def q_ref_dq_to_fbgemm(ctx: GraphModContext) -> None:
    # key: matched q/dq node, value: q/dq's consumers to be moved up
    qdq_nodes: Dict[Node, Set[Node]] = defaultdict(set)
    # TODO(litan.ls): refactor this to generic pattern matching func
    # pattern match dq+ref_quant+q -> real_quant
    for q_node in ctx.nodes_by_function_type([torch.quantize_per_tensor]):

        LOGGER.debug(f'q_node: {q_node.target}')
        ref_node = q_node.args[0]
        LOGGER.debug(f'ref_node: {ref_node.target}')
        ref = ctx.modules.get(ref_node.target)
        if type(ref) not in REF_TO_QUANT_MAP:
            continue
        dq_node = ref_node.args[0]
        LOGGER.debug(f'dq_node: {dq_node.target}')
        if dq_node.op != 'call_method' or dq_node.target != 'dequantize':
            continue
        dst_type = REF_TO_QUANT_MAP[type(ref)]
        scale = ctx.get_attr(q_node.args[1].target)
        zero_point = ctx.get_attr(q_node.args[2].target)
        dst = dst_type.from_reference(ref, scale, zero_point)
        ctx.replace_module(ref_node.target, dst)
        qdq_nodes[q_node].update(q_node.users)
        qdq_nodes[dq_node].add(ref_node)
        LOGGER.debug(
            f'ref_to_fbgemm: {ref_node.target}({type(ref)}->{dst_type})')
    for node, users in qdq_nodes.items():
        node.replace_all_uses_with(
            node.args[0], delete_user_cb=lambda user: user in users)
    ctx.gm.graph.eliminate_dead_code()
    ctx.gm.recompile()


def fold_qdq(ctx: GraphModContext) -> None:
    # TODO(litan.ls): refactor this to generic pattern matching func
    for dq in ctx.nodes_by_method_names(['dequantize']):
        q = dq.args[0]
        if q.op == 'call_function' and q.target in [torch.quantize_per_tensor]:
            dq.replace_all_uses_with(q.args[0])
            ctx.gm.graph.erase_node(dq)
            ctx.gm.graph.erase_node(q)
    ctx.gm.graph.eliminate_dead_code()
    ctx.gm.recompile()

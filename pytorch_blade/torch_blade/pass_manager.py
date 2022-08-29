# Copyright 2021 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch_blade
from torch.onnx import OperatorExportTypes
from torch.onnx.symbolic_helper import _set_opset_version
from torch_blade import tools, utils
from torch_blade.config import Config
from torch_blade.python_ir_analysis import _jit_pass_clean_python_ir
from torch_blade.quantization import (
    _jit_pass_quantization_postprocess,
    _jit_pass_quantization_preprocess
)


def _get_current_onnx_opset_version():
    if utils.torch_version_number() < utils.parse_version("1.12.0"):
        from torch.onnx.symbolic_helper import _export_onnx_opset_version
    else:
        from torch.onnx._globals import GLOBALS
        _export_onnx_opset_version = GLOBALS.export_onnx_opset_version
    return _export_onnx_opset_version


# tools are some function borrowed from torch that is private,
# which should be use carefully

# TODO(gty):
# This option could be removed once we fully support static analysis of shape and rank.
# Also to avoid change on API, we would set the option through a global variable.
IGNORE_DYNAMIC_RANK = True


def _get_dynamic_axes(input_list, dynamic_axes):
    dyn_axs = {}
    if dynamic_axes is None:
        return dyn_axs

    for inp, axes_ds in zip(input_list, dynamic_axes):
        name = inp.debugName()
        axes = {}
        for idx, d in enumerate(inp.type().sizes()):
            if axes_ds[idx] == d:
                continue
            axes[idx] = "{}_axis_{}".format(name, idx)
        dyn_axs[name] = axes
    return dyn_axs


def _set_opset_version_from_config():
    """
    Consulting cfg.customize_onnx_opset_version and set global ONNX opset
    version. And return the opset version.
    """
    cfg = Config.get_current_context_or_new()
    if cfg.customize_onnx_opset_version:
        opset_version = cfg.customize_onnx_opset_version
        _set_opset_version(opset_version)
        return opset_version
    return _get_current_onnx_opset_version()


def _export_onnx(graph, dynamic_axes, fold_constants=True):
    # Note: TRT7 support opset 11
    opset_version = _set_opset_version_from_config()

    if fold_constants:
        graph, params_dict = _jit_pass_onnx_constfold(graph, {})
    else:
        params_dict = {}

    dynamic_axes = _get_dynamic_axes(graph.input_list(), dynamic_axes)
    defer_weight_export = False
    operator_export_type = OperatorExportTypes.ONNX
    strip_doc_string = True
    val_keep_init_as_ip = False
    custom_opsets = {}
    val_add_node_names = True
    proto = graph._export_onnx(
        params_dict,
        opset_version,
        dynamic_axes,
        defer_weight_export,
        operator_export_type,
        strip_doc_string,
        val_keep_init_as_ip,
        custom_opsets,
        val_add_node_names,
    )[0]
    return proto


def _jit_pass_dce_during_lower_to_trt(graph):
    torch._C._jit_pass_dce(graph)


def _jit_pass_lint(graph):
    torch._C._jit_pass_lint(graph)


def _jit_pass_lower_to_onnx(graph):
    """ currently _jit_pass_lower_all_tuples will modified the graph output if it's tuple"""
    # NB(xiafei.qiuxf): Should set opset version here to be consistent with _export_onnx.
    _set_opset_version_from_config()
    current_onnx_opset_version = _get_current_onnx_opset_version()
    # onnx does not support tuples, so try to remove them
    # torch._C._jit_pass_lower_all_tuples(graph)
    # torch._C._jit_pass_peephole(graph, True)
    # torch._C._jit_pass_lint(graph)

    # onnx only supports tensors, but 1 / 2 = 0.5 and tensor(1) / tensor(2) = 0
    torch._C._jit_pass_prepare_division_for_onnx(graph)

    torch._C._jit_pass_onnx_remove_print(graph)

    torch._C._jit_pass_onnx_preprocess_caffe2(graph)

    # onnx only supports tensors, so we turn all out number types into tensors
    torch._C._jit_pass_erase_number_types(graph)

    # Should update torch_blade.tools._jit_pass_onnx to
    # https://github.com/pytorch/pytorch/blob/v1.11.0/torch/csrc/jit/passes/onnx.cpp#L167
    #
    # But something weird happened:
    # call to ConstantValueMap::ClearMaps() in C++ will segfault
    if utils.torch_version_number() >= utils.parse_version("1.9.0"):
        onnx_graph = torch._C._jit_pass_onnx(graph, OperatorExportTypes.ONNX)
        # fix(tanyo): call to torch_blade.tools._jit_pass_onnx would segfault
        value_map = dict()
    else:
        onnx_graph, value_map = torch_blade.tools._jit_pass_onnx(
            graph, OperatorExportTypes.ONNX
        )

    # extract debugName to avoid crash (caused by jit_pass which may modify jit.Value)
    value_map = {
        t_value.debugName(): o_value.debugName()
        for t_value, o_value in value_map.items()
    }

    # pylint: disable=maybe-no-member
    torch_blade.jit_pass_onnx_constant_f64_to_f32(onnx_graph)

    # lint the graph
    torch._C._jit_pass_lint(onnx_graph)

    if utils.torch_version_number() >= utils.parse_version("1.9.0"):
        torch._C._jit_pass_onnx_scalar_type_analysis(onnx_graph, True, current_onnx_opset_version)
    else:
        torch._C._jit_pass_onnx_scalar_type_analysis(onnx_graph)
    torch._C._jit_pass_lint(onnx_graph)

    torch._C._jit_pass_onnx_peephole(onnx_graph, current_onnx_opset_version, False)
    torch._C._jit_pass_lint(onnx_graph)

    # graph is not a valid jit graph anymore because types have been replaced
    # (e.g. int with Tensor), so it now contains operators that don't actually
    # exist. We can't run normal dead code elimination because it'd fail trying
    # to look up if an operator has side effects, but we can run a dead code
    # elimination variant that doesn't need to look up if an op has side effects.
    torch._C._jit_pass_dce_allow_deleting_nodes_with_side_effects(onnx_graph)
    torch._C._jit_pass_lint(onnx_graph)
    graph = torch._C._jit_pass_canonicalize(onnx_graph)
    torch._C._jit_pass_lint(onnx_graph)

    return onnx_graph, value_map


def _jit_pass_onnx_constfold(graph, params_dict):
    current_onnx_opset_version = _get_current_onnx_opset_version()
    if current_onnx_opset_version >= 9:
        params_dict = torch._C._jit_pass_onnx_constant_fold(
            graph, params_dict, current_onnx_opset_version
        )
        torch._C._jit_pass_dce_allow_deleting_nodes_with_side_effects(graph)
    torch._C._jit_pass_lint(graph)
    return graph, params_dict


def _jit_pass_freeze_rank(graph):
    def freeze_rank_analysis(outer_block):
        dim_nodes = [
            n
            for n in outer_block.nodes()
            if n.kind() == "aten::dim" and n.input().isCompleteTensor()
        ]
        for node in dim_nodes:
            assert len(node.input_list()) == 1
            rank = node.input().type().dim()
            val = graph.insertConstant(rank)
            val.node().moveAfter(node)
            node.output().replaceAllUsesWith(val)

        for node in outer_block.nodes():
            for blk in node.blocks():
                freeze_rank_analysis(blk)

    freeze_rank_analysis(graph)
    torch._C._jit_pass_dce(graph)
    torch._C._jit_pass_constant_propagation(graph)

def _jit_pass_freeze_requires_grad(graph):
    # Note: this replace requires_grad to false,
    # should only be used at eval stage
    from_graph_str = """
    graph(%x):
      %r : bool = prim::requires_grad(%x)
      return (%r)
    """
    to_graph_str = """
    graph(%x):
      %r : bool = prim::Constant[value=0]()
      return (%r)
    """
    torch._C._jit_pass_custom_pattern_based_rewrite_graph(
        from_graph_str, to_graph_str, graph)
    torch._C._jit_pass_dce(graph)
    torch._C._jit_pass_constant_propagation(graph)

def _jit_pass_reorder_raise_exception(graph):
    def _reorder_raise_exception_node(graph):
        graph_nodes = graph.node_list()
        num_nodes = len(graph_nodes)
        for i, node in enumerate(graph_nodes):
            # the following ifs are to identify the IR of this sort:
            # if ?:
            #     prim.RaiseException("Exception")
            # else:
            #     pass
            if (
                node.kind() == "prim::If"
                and len(node.output_list()) == 0
                and len(node.input_list()) == 1
            ):
                blocks = [blk for blk in node.blocks()]
                if (
                    len(blocks) == 2
                    and len(blocks[0].node_list()) == 1
                    and blocks[0].node_list()[0].kind() == "prim::RaiseException"
                    and len(blocks[1].node_list()) == 0
                ):
                    # if pattern matched, then move the if-raise-exception
                    # towards the end of graph as further as we can
                    node_input = node.input_list()[0]
                    for j in range(i + 1, num_nodes):
                        # node_input is found in the current node inputs
                        # if move if-raise-exception further of this point will
                        # break the topology order thus move it after the previous node
                        if node_input in graph_nodes[j].input_list() and j != i + 1:
                            node.moveAfter(graph_nodes[j - 1])
                            return node_input
                        # reach the end of the graph
                        # if-raise-exception can be move safely here
                        if j == num_nodes - 1:
                            node.moveAfter(graph_nodes[j])
                            return node_input
        return None

    processed = list()
    while True:
        mark = _reorder_raise_exception_node(graph)
        if mark in processed:
            break
        else:
            processed.append(mark)
    torch._C._jit_pass_dce(graph)


def _jit_pass_remove_nograd(graph):
    def remove_no_grad(outer_block):
        node_list = [n for n in outer_block.nodes()]
        grad_block_nodes = []
        for node in node_list:
            if node.kind() == 'prim::CreateObject' and "torch.autograd.grad_mode.no_grad" in str(node.output().type()):
                users = [u.user for u in node.output().uses()]
                grad_block_nodes.append(node)
                grad_block_nodes += users
                continue
            for blk in node.blocks():
                remove_no_grad(blk)
        for node in reversed(grad_block_nodes):
            node.destroy()
    remove_no_grad(graph)
    torch._C._jit_pass_dce(graph)


def _jit_pass_clean_script(graph):
    def remove_raise_exception(outer_block):
        node_list = [n for n in outer_block.nodes()]
        for node in node_list:
            if node.kind() == 'prim::RaiseException':
                node.destroy()
                continue
            for blk in node.blocks():
                remove_raise_exception(blk)

    remove_raise_exception(graph)
    torch._C._jit_pass_dce(graph)


def _optimize_common(c_module, static_shape=False):
    _jit_pass_quantization_preprocess(c_module)
    is_training = c_module.hasattr("training") and c_module.training
    if not is_training:
        # optimization passes only work in eval mode
        cfg = Config.get_current_context_or_new()
        presv_attrs = cfg.preserved_attributes
        c_module = tools.freeze_module(c_module, presv_attrs, disableShapePeephole=not static_shape)
        torch._C._jit_pass_remove_dropout(c_module)
        graph = c_module.forward.graph
        _jit_pass_remove_nograd(graph)
        _jit_pass_freeze_requires_grad(graph)
        if hasattr(torch._C, "_jit_pass_fold_frozen_conv_bn"):
            torch._C._jit_pass_fold_frozen_conv_bn(graph)

    graph = c_module.forward.graph
    torch._C._jit_pass_remove_mutation(graph)

    # TODO: if dynamic rank exists, this pass maybe leads to error
    if IGNORE_DYNAMIC_RANK:
        _jit_pass_freeze_rank(c_module.forward.graph)

    tools._jit_pass_lower_simple_tuples(c_module.forward.graph)
    _jit_pass_clean_script(c_module.forward.graph)

    # The _jit_pass_clean_python_ir was placed here,
    # because it needs some preprocess jit pass before,
    # such as remove grads ir nodes, freeze rank, tuple lowering etc.
    _jit_pass_clean_python_ir(graph)
    _jit_pass_quantization_postprocess(c_module)
    return c_module


def _jit_pass_licm(graph):
    torch._C._jit_pass_remove_mutation(graph)
    tools.licm(graph)

def _jit_pass_patine_conv2d(graph):
    torch._C._jit_pass_custom_pattern_based_rewrite_graph("""
graph(%input, %weight, %bias, %stride, %padding, %dilation, %group):
   %r = aten::conv2d(%input, %weight, %bias, %stride, %padding, %dilation, %group)
   %r = aten::relu(%r)
   return (%r)""", """
graph(%input, %weight, %bias, %stride, %padding, %dilation, %group):
   %0 : int = prim::Constant[value=0]()
   %1 : int = prim::Constant[value=1]()
   %2 : int = prim::Constant[value=2]()
   %3 : int = prim::Constant[value=3]()
   %nhwc_dims : int[] = prim::ListConstruct(%0, %2, %3, %1)
   %nhwc = aten::permute(%input, %nhwc_dims)
   %r = patine::conv2d_relu_nhwc(%nhwc, %weight, %bias, %stride, %padding, %dilation, %group)
   %nchw_dims : int[] = prim::ListConstruct(%0, %3, %1, %2)
   %nchw = aten::permute(%r, %nchw_dims)
   return (%nchw)""", graph)
    tools.eliminate_redundant_permutations(graph)

def _jit_pass_hack_cpu_device(graph):
    cfg = Config.get_current_context_or_new()
    if not cfg.enable_force_to_cuda:
        return

    torch._C._jit_pass_inline(graph)
    nodes = [n for n in graph.nodes() if 'prim::Constant' in n.kind()]
    for prim_const in nodes:
        if prim_const.hasAttribute('value') and prim_const.kindOf('value') == 's' and prim_const.s('value') == 'cpu':
            prim_const.s_('value', 'cuda')


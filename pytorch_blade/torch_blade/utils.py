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

import os
import collections
import contextlib
import functools
import torch
from torch_blade.algorithm import NxGraph


@contextlib.contextmanager
def cwd(cur_dir):
    old_dir = os.getcwd()
    try:
        os.chdir(os.path.expanduser(cur_dir))
        yield
    finally:
        os.chdir(old_dir)


def add_method(cls, name=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        setattr(cls, name or func.__name__, wrapper)
    return decorator


def find_control_dependencies(node: torch._C.Node):
    control_deps = []
    for block in node.blocks():
        # collect values defined in current block
        vals_in_block = node.input_list() + [inp for inp in block.inputs()]
        for n in block.nodes():
            vals_in_block += n.output_list()
        vals_in_block = set(vals_in_block)
        all_deps = []
        for n in block.nodes():
            deps = find_control_dependencies(n)
            deps += n.input_list()
            all_deps += deps
        all_deps += block.returnNode().input_list()
        all_deps = set(all_deps)
        control_deps += [dep for dep in all_deps if dep not in vals_in_block]
    return list(set(control_deps))


def graph_in_topology_order(graph):
    value_set = set()

    for inp in graph.inputs():
        value_set.add(inp.debugName())

    for node in graph.node_list():
        input_deps = node.input_list() + node.control_deps()
        for inp in input_deps:
            if inp.debugName() not in value_set:
                return False
        for out in node.outputs():
            value_set.add(out.debugName())

    for out in graph.outputs():
        if out.debugName() not in value_set:
            return False
    return True

def build_nxgraph_of_nodes(block):
    nodes = block.node_list()
    node2idx = dict([(n, idx) for (idx, n) in enumerate(nodes)])
    nx_graph = NxGraph()

    control_nodes = []
    for idx, node in enumerate(nodes):
        input_deps = node.input_list() + node.control_deps()
        inputs = [node2idx[inp.node()]
                  for inp in input_deps if inp.node() in node2idx]
        inputs = set(inputs)
        nx_graph.add_node(idx)
        for src in inputs:
            nx_graph.add_edge(src, idx)

        # TODO(fix): To workaround ASR model compilation.
        # Please relax the dependencies.
        for ctl_n in control_nodes:
            nx_graph.add_edge(node2idx[ctl_n], idx)

        if len(list(node.blocks())) > 0:
            control_nodes.append(node)

    return nx_graph

def block_topology_ajust(block):
    nodes = block.node_list()
    if (len(nodes) <= 1):
        return
    nx_graph = build_nxgraph_of_nodes(block)
    topo_idx = nx_graph.lexical_order_topolist()
    assert (len(topo_idx) == len(nodes))

    node0 = nodes[topo_idx[0]]
    if (node0 != nodes[0]):
        node0.moveBefore(nodes[0])

    prev_node = node0
    topo_idx.pop(0)
    for idx in topo_idx:
        next_node = nodes[idx]
        next_node.moveAfter(prev_node)
        prev_node = next_node

def listify(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        return list(func(*args, **kwargs))
    return wrapped


def graph_node_topolist(graph: torch._C.Graph):
    assert(graph_in_topology_order(graph))
    return graph.node_list()


def list_prim_ops(graph):
    count = collections.Counter()
    for node in graph.nodes():
        if (node.kind().startswith('prim::')):
            count[node.kind()] += 1
    return count


def list_shape_ops(graph):
    count = collections.Counter()
    for node in graph.nodes():
        if ('shape' in node.kind()):
            count[node.kind()] += 1
    return count


def list_can_constfold(graph):
    count = collections.Counter()
    for node in graph.nodes():
        can_constfold = True
        for inp in node.inputs():
            is_const = inp.node().kind() == 'prim::Constant'
            if (not is_const):
                can_constfold = False

        if (can_constfold):
            count[node.kind()] += 1
    return count


def list_ops_count(graph):
    count = collections.Counter()
    for node in graph.nodes():
        count[node.kind()] += 1
    return count


def collect_engines(script_module, group_type):
    """
    Collect all engines in a script_module of the group type
    """
    def collect_engines_in_block(block):
        engine_names = [
            n.s('name') for n in block.nodes()
            if n.kind() == 'prim::GetAttr' and n.s('name').startswith(group_type)
        ]
        for n in block.nodes():
            for blk in n.blocks():
                engine_names += collect_engines_in_block(blk)
        return engine_names

    if isinstance(script_module, torch.jit.ScriptModule):
        script_module = script_module._c
    engine_names = collect_engines_in_block(script_module.forward.graph)
    engines = [script_module.getattr(name) for name in engine_names]
    return list(zip(engine_names, engines))

def num_engines(script_module, group_type):
    """
    Return the number of engines of the group_type
    """
    return len(collect_engines(script_module, group_type))


def num_compiled_nodes(script_module, group_type):
    """
    Return the number of compiled nodes of the group_type
    """
    import re
    # pattern like 'disc/trt_grp*_len*_*'
    p = re.compile(r'(?<=len)\d+')
    return [int(p.search(s).group(0)) for s, _ in collect_engines(script_module, group_type)]


def parse_version(version: str):
    """
    Parses a version string into (major, minor, patch) version numbers.

    Args:
      version: Full version number string, possibly including revision / commit hash.

    Returns:
      An int 3-tuple of (major, minor, patch) version numbers.
    """
    # Extract version number part (i.e. toss any revision / hash parts).
    version_number_str = version
    for i in range(len(version)):
        c = version[i]
        if not (c.isdigit() or c == "."):
            version_number_str = version[:i]
            break

    return tuple([int(n) for n in version_number_str.split(".")[:3]])

def torch_version_number():
    return parse_version(torch.__version__)

def create_list_construct(graph, vals, list_type):
    list_ctr = graph.create('prim::ListConstruct')
    for val in vals:
        list_ctr.addInput(val)
    list_ctr.output().setType(list_type)
    graph.appendNode(list_ctr)
    return list_ctr

def subgraph_to_module(subgraph, group_name):
    for idx, inp in enumerate(subgraph.inputs()):
        # input debguNames like input.1_ are not allowed in python code,
        # also debugNames consist of numbers, such as '1111', are not allowed
        inp.setDebugName("inp_%s" % idx)
    ret_node = subgraph.return_node()
    ret_vals = ret_node.input_list()
    # make fallback return a List[Tensor]
    assert(all(isinstance(val.type(), torch._C.TensorType) for val in ret_vals))
    list_ctr_type = torch._C.ListType(torch._C.TensorType.get())
    list_ctr = create_list_construct(subgraph, ret_vals, list_ctr_type)
    ret_node.removeAllInputs()
    ret_node.addInput(list_ctr.output())

    _compilation_unit = torch._C.CompilationUnit()
    fallback_module = torch._C.ScriptModule(group_name, _compilation_unit, True)
    fallback_module.create_method_from_graph('forward', subgraph)
    return fallback_module

def dump_graph_and_code(module, fpath):
    # Note: this is develop & debug utils
    with open(fpath + ".graph.txt", "w") as out:
        out.write(str(module.forward.graph))

    with open(fpath + ".code.py", "w") as out:
        out.write(str(module.forward.code))

def disable_pytorch_jit():
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(False)

def _inline_node_with_subgraph(graph, old_node, subgraph):
    value_map = dict()
    for sbg_inp, inp in zip(subgraph.inputs(), old_node.inputs()):
        value_map[sbg_inp] = inp

    for sbg_node in graph_node_topolist(subgraph):
        new_node = graph.createClone(sbg_node, lambda x: value_map[x])
        # append new node to graph
        graph.appendNode(new_node)
        new_node.moveBefore(old_node)
        for sbg_out, out in zip(sbg_node.outputs(), new_node.outputs()):
            value_map[sbg_out] = out
    outputs = []
    for sbg_out, grp_out in zip(subgraph.outputs(), old_node.outputs()):
        grp_out.replaceAllUsesWith(value_map[sbg_out])
        outputs.append(value_map[sbg_out])

    old_node.destroy()
    return outputs

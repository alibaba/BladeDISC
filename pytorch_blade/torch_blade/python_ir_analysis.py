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

from collections import namedtuple

import torch

from torch_blade import tools
from torch_blade import utils

Container = namedtuple("Container", ["instance", "static"])

# TODO(gty): Refactor this module

def _is_primitive(val):
    return type(val) in [str, int, bool]

def _update_list_container(container_map, local, graph, node, val, container):
    list_ctr = utils.create_list_construct(graph, container.instance, val.type())
    list_ctr.moveAfter(node)
    val.replaceAllUsesWith(list_ctr.output())
    container_map[list_ctr.output()] = container
    local.add(list_ctr.output())

def _copy_list_container(container_map, local, graph, node, val, container):
    list_ctr = utils.create_list_construct(graph, container.instance, val.type())
    list_ctr.moveBefore(node)
    node.replaceInputWith(val, list_ctr.output())
    container_map[list_ctr.output()] = container
    local.add(list_ctr.output())

def _prim_dict_construct(container_map, local, graph, node):
    dc_inputs = node.input_list()
    keys = dc_inputs[::2]
    all_const = all(k.node().kind() == "prim::Constant" for k in keys)
    if not all_const:
        return
    keys = [k.toIValue() for k in keys]

    all_primitive = all(_is_primitive(k) for k in keys)
    if not all_primitive:
        return
    vals = dc_inputs[1::2]
    dc_inputs = dict(zip(keys, vals))
    container_map[node.output()] = Container(instance=dc_inputs, static=True)
    local.add(node.output())

def _prim_constant(container_map, local, graph, node):
    if str(node.output().type()).startswith("Dict"):
        dict_instance = node.output().toIValue()
        vals = dict_instance.values()
        vals = [graph.insertConstant(v) for v in vals]
        for v in reversed(vals):
            v.node().moveAfter(node)
        dc_inputs = dict(zip(dict_instance.keys(), vals))
        container_map[node.output()] = Container(instance=dc_inputs, static=True)
    elif str(node.output().type()).startswith("List"):
        vals = [graph.insertConstant(v) for v in node.output().toIValue()]
        for v in reversed(vals):
            v.node().moveAfter(node)
        container_map[node.output()] = Container(instance=vals, static=True)

def _prim_list_construct(container_map, local, graph, node):
    lc_inputs = node.input_list()
    container_map[node.output()] = Container(instance=lc_inputs, static=True)
    local.add(node.output())

def _aten_set_item(container_map, local, graph, node):
    inp0, inp1, inp2 = node.input_list()
    container = container_map.get(inp0, None)
    do_static_analysis = container is not None and container.static
    if not do_static_analysis:
        return
    if inp0 not in local or inp1.node().kind() != "prim::Constant":
        # the modify is dynamic, stop static analysis
        container_map[inp0] = Container(instance=container.instance, static=False)
    elif isinstance(container.instance, dict):
        idx = inp1.toIValue()
        container.instance[idx] = inp2
    elif isinstance(container.instance, list):
        idx = inp1.toIValue()
        if idx < len(container.instance):
            # warn: assert idx < len(container.instance)
            container.instance[idx] = inp2
            _update_list_container(container_map, local, graph, node, inp0, container)
            node.destroy()
        else:
            # to ensure we are not error-prone, we disable the static analysis
            container_map[inp0] = Container(instance=container.instance, static=False)

def _aten_get_item(container_map, local, graph, node):
    inp0, inp1 = node.input_list()
    container = container_map.get(inp0, None)
    do_static_analysis = container is not None and container.static
    if not do_static_analysis:
        return
    if inp1.node().kind() == "prim::Constant":
        idx = inp1.toIValue()
        if isinstance(container.instance, dict) and idx in container.instance:
            item = container.instance[idx]
            node.output().replaceAllUsesWith(item)
            node.destroy()
            return
        elif isinstance(container.instance, list) and idx < len(container.instance):
            item = container.instance[idx]
            node.output().replaceAllUsesWith(item)
            node.destroy()
            return

    # During python ir analysis the list/dict aten nodes would be modify inplace.
    # To avoid data hazard, we would like to copy the list or stop static analysis.
    if isinstance(container.instance, list):
        _copy_list_container(container_map, local, graph, node, inp0, container)
    else:
        # to ensure we are not error-prone, we disable the static analysis
        container_map[inp0] = Container(instance=container.instance, static=False)

def _aten_contains(container_map, local, graph, node):
    inp0, inp1 = node.input_list()
    container = container_map.get(inp0, None)
    do_static_analysis = container is not None and container.static
    if not do_static_analysis:
        return
    if inp1.node().kind() == "prim::Constant":
        idx = inp1.toIValue()
        if not _is_primitive(idx):
            # TODO: currently only support aten::__contains__ of primitive types
            # aten::__contains__.str(Dict(str, t) dict, str key) -> (bool):
            # aten::__contains__.int(Dict(int, t) dict, int key) -> (bool):
            # aten::__contains__.float(Dict(float, t) dict, float key) -> (bool):
            # aten::__contains__.int(int[] l, int item) -> (bool):
            # aten::__contains__.float(float[] l, float item) -> (bool):
            # aten::__contains__.str(str[] l, str item) -> (bool):
            #
            # The following schema is not supported so far,
            # aten::__contains__.Tensor(Dict(Tensor, t) dict, Tensor key) -> (bool):
            return
        if isinstance(container.instance, list):
            elems = [v.toIValue() for v in container.instance]
        else:
            elems = container.instance
        if idx in elems:
            val = graph.insertConstant(True)
        else:
            val = graph.insertConstant(False)
        val.node().moveAfter(node)
        node.output().replaceAllUsesWith(val)
        node.destroy()
        return

    # During python ir analysis the list/dict aten nodes would be modify inplace.
    # To avoid data hazard, we would like to copy the list or stop static analysis.
    if isinstance(container.instance, list):
        _copy_list_container(container_map, local, graph, node, inp0, container)
    else:
        # to ensure we are not error-prone, we disable the static analysis
        container_map[inp0] = Container(instance=container.instance, static=False)

def _aten_append(container_map, local, graph, node):
    inp0, inp1 = node.input_list()
    container = container_map.get(inp0, None)
    do_static_analysis = container is not None and container.static
    if not do_static_analysis:
        return
    if isinstance(container.instance, dict):
        # TODO: undefined, stop static analysis
        container_map[inp0] = Container(instance=container.instance, static=False)
    elif isinstance(container.instance, list):
        if inp0 in local:
            container.instance.append(inp1)
            _update_list_container(container_map, local, graph, node, inp0, container)
            node.destroy()
        else:
            container_map[inp0] = Container(instance=container.instance, static=False)

def _aten_extend(container_map, local, graph, node):
    inp0, inp1 = node.input_list()
    container0 = container_map.get(inp0, None)
    container1 = container_map.get(inp1, None)
    do_static_analysis = container0 is not None and container0.static
    do_static_analysis = do_static_analysis and container1 is not None and container1.static
    if do_static_analysis and isinstance(container0.instance, list) and isinstance(container1.instance, list):
        if inp0 in local:
            container0.instance.extend(container1.instance)
            _update_list_container(container_map, local, graph, node, inp0, container0)
            node.destroy()
            return
    # TODO: undefined, stop static analysis
    container_map[inp0] = Container(instance=container0.instance, static=False)

def _aten_insert(container_map, local, graph, node):
    inp0, inp1, inp2 = node.input_list()
    container = container_map.get(inp0, None)
    do_static_analysis = container is not None and container.static
    if not do_static_analysis:
        return
    if isinstance(container.instance, dict):
        # TODO: undefined, stop static analysis
        container_map[inp0] = Container(instance=container.instance, static=False)
    elif isinstance(container.instance, list):
        if inp0 not in local:
            container_map[inp0] = Container(instance=container.instance, static=False)
        elif inp1.node().kind() != "prim::Constant":
            container_map[inp0] = Container(instance=container.instance, static=False)
        else:
            idx = inp1.toIValue()
            container.instance.insert(idx, inp2)
            _update_list_container(container_map, local, graph, node, inp0, container)
            node.destroy()

def _aten_keys(container_map, local, graph, node):
    (inp0,) = node.input_list()
    container = container_map.get(inp0, None)
    do_static_analysis = container is not None and container.static
    if do_static_analysis and isinstance(container.instance, dict):
        keys = container.instance.keys()
        vals = [graph.insertConstant(v) for v in keys]
        list_container = Container(instance=vals, static=True)
        _update_list_container(container_map, local, graph, node, node.output(), list_container)
        node.destroy()

    # TODO: undefined, stop static analysis
    container_map[inp0] = Container(instance=container.instance, static=False)

def _aten_len(container_map, local, graph, node):
    (inp0,) = node.input_list()
    container = container_map.get(inp0, None)
    do_static_analysis = container is not None and container.static
    if not do_static_analysis:
        return
    val = graph.insertConstant(len(container.instance))
    val.node().moveAfter(node)
    node.output().replaceAllUsesWith(val)
    node.destroy()

def _aten_undefined(container_map, local, graph, node):
    for inp in node.inputs():
        if inp not in container_map:
            continue
        container = container_map[inp]
        # meet undefined use, stop static analysis
        container_map[inp] = Container(instance=container.instance, static=False)

def _jit_pass_clean_python_ir(graph, is_training=False):
    handler = {
        "prim::DictConstruct": _prim_dict_construct,
        "prim::ListConstruct": _prim_list_construct,
        "prim::Constant": _prim_constant,
        "aten::_set_item": _aten_set_item,
        "aten::__getitem__": _aten_get_item,
        "aten::append": _aten_append,
        "aten::insert": _aten_insert,
        "aten::len": _aten_len,
        "aten::__contains__": _aten_contains,
        "aten::extend": _aten_extend,
        "aten::keys": _aten_keys,
    }
    container_map = dict()

    def analysis_python_ir(outer_block):
        # This is actually a simulator of TorchScript operations.
        # We would tape the value evaluated during the simulaton,
        # and replace the usage of the original value to constant if needed.
        # After that we would do dce & constant propagation.
        #
        # An operation's behavior is undefined if there is no handler for it.
        # In this case, to ensure the simulator work correctly,
        # we would like to stop simulation on corresponding container value instance.

        local = set()  # local set is used to mark block scope.
        node_list = [n for n in outer_block.nodes()]
        for node in node_list:
            if node.kind() in handler:
                handler[node.kind()](container_map, local, graph, node)
            else:
                _aten_undefined(container_map, local, graph, node)
                for blk in node.blocks():
                    analysis_python_ir(blk)
        for inp in local:
            del container_map[inp]

    for idx in range(0, 10):
        # iterate several times to saturate the analysis
        torch._C._jit_pass_dce(graph)
        tools._jit_pass_lower_simple_tuples(graph)
        tools._jit_pass_const_loop_unrolling(graph)
        if not is_training:
            # training with dynamo 
            torch._C._jit_pass_constant_propagation(graph)

        analysis_python_ir(graph)
    # eliminate dead codes create during analysis_python_ir
    torch._C._jit_pass_dce(graph)

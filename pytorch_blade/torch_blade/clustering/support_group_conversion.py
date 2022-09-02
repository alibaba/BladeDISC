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
from torch_blade import pass_manager, tensorrt, tools, utils
from torch_blade.logging import logger


def _inline_node_with_subgraph(graph, old_node, subgraph):
    value_map = dict()
    for sbg_inp, inp in zip(subgraph.inputs(), old_node.inputs()):
        value_map[sbg_inp] = inp

    for sbg_node in utils.graph_node_topolist(subgraph):
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


def replace_group_with_engine(
        graph,
        module_holder,
        node,
        attr_name,
        eng_type,
        group_inputs=True,
        engine_method_name="execute",
):
    attr = graph.create('prim::GetAttr')
    attr.addInput(module_holder)
    attr.s_('name', attr_name)
    attr.output().setType(eng_type)
    graph.appendNode(attr)
    attr.moveBefore(node)

    if group_inputs:
        # create input_list of self.engine.execute, something like:
        # %12 : Tensor[] = prim::ListConstruct(%10, %11)
        list_constuct = graph.create('prim::ListConstruct')
        for inp in node.inputs():
            list_constuct.addInput(inp)
        list_constuct.output().setType(torch_blade.tools.get_list_tensor_type())
        graph.appendNode(list_constuct)
        list_constuct.moveBefore(node)

    # create prim::CallMethod, something like:
    call_method = graph.create('prim::CallMethod')
    call_method.s_('name', engine_method_name)
    call_method.addInput(attr.output())
    if group_inputs:
        # %5 : Tensor[] = prim::CallMethod[name="execute"](%3, %input_list)
        call_method.addInput(list_constuct.output())
    else:
        # %5 : Tensor[] = prim::CallMethod[name="execute"](%3, %input1, %input2, ...)
        for inp in node.input_list():
            call_method.addInput(inp)

    call_method.output().setType(torch_blade.tools.get_list_tensor_type())
    graph.appendNode(call_method)
    call_method.moveBefore(node)

    # create prim::ListUnpack, something like:
    # %17 : Tensor, %18 : Tensor, %19 : Tensor = prim::ListUnpack(%16)
    list_unpack = graph.create('prim::ListUnpack')
    list_unpack.addInput(call_method.output())
    list_unpack.eraseOutput(0)
    graph.appendNode(list_unpack)
    list_unpack.moveBefore(node)

    for out in node.outputs():
        lu_out = list_unpack.addOutput()
        lu_out.setType(out.type())
        out.replaceAllUsesWith(lu_out)

    # destory node
    node.destroy()
    return attr


def _adapt_input_value(graph, consumer_node, value):
    placeholder = graph.create('torch_blade::placeholder')
    placeholder.addInput(value)
    placeholder.output().setType(value.type())
    graph.appendNode(placeholder)
    placeholder.moveBefore(consumer_node)
    tools.cast_to_tensor_type(placeholder.output())
    consumer_node.replaceInputWith(value, placeholder.output())

    gstr = """
    graph(%x.1 : int):
      %4 : bool = prim::Constant[value=0]()
      %2 : None = prim::Constant()
      %5 : Long() = aten::tensor(%x.1, %2, %2, %4)
      return (%5)"""
    subgraph = torch._C.parse_ir(gstr)
    _inline_node_with_subgraph(graph, placeholder, subgraph)


def _adapt_output_value(graph, producer_node, value):
    placeholder = graph.create('torch_blade::placeholder')
    value.replaceAllUsesWith(placeholder.output())
    placeholder.addInput(value)
    placeholder.output().setType(value.type())
    graph.appendNode(placeholder)
    assert(producer_node == value.node())
    placeholder.moveAfter(producer_node)
    tools.cast_to_tensor_type(value)

    # TODO: aten::Int would overflow if input > 2^31 - 1
    gstr = """
    graph(%x.1 : Long()):
      %2 : Scalar = aten::item(%x.1)
      %3 : int = aten::Int(%2)
      return (%3)"""
    subgraph = torch._C.parse_ir(gstr)
    _inline_node_with_subgraph(graph, placeholder, subgraph)


def _is_number(val):
    return val.type().isSubtypeOf(torch._C.NumberType.get())


def _adapt_node_number_inputs(graph, node):
    number_inps = [(idx, inp) for idx, inp in enumerate(node.inputs()) if _is_number(inp)]
    for idx, inp in number_inps:
        _adapt_input_value(graph, node, inp)


def _adapt_node_number_outputs(graph, node):
    number_outs = [(idx, out) for idx, out in enumerate(node.outputs()) if _is_number(out)]
    for idx, out in number_outs:
        _adapt_output_value(graph, node, out)


def group_node_to_engine(
        module, node,
        try_cvt_to_engine_func,
        adapt_number_ios,
        idxes,
        grp_calib_data=None
):
    if (node.kind() != 'prim::FusionGroup'):
        return

    if (len(node.input_list()) == 0):
        logger.warning('The prim::FusionGroup has no input is not allowed')
        return

    # NB: use subgraph copy because we might modify the subgraph
    fallback_subgraph = node.g('Subgraph')
    subgraph = fallback_subgraph.copy()

    group_idx = idxes[0]
    subgraph_idx = idxes[1]
    num_nodes = len(subgraph.node_list())
    group_name = "%s_len%s_%s" % (group_idx, num_nodes, subgraph_idx)

    if adapt_number_ios:
        # NB: Would modify the subgraph
        _adapt_node_number_outputs(subgraph, subgraph.param_node())
        _adapt_node_number_inputs(subgraph, subgraph.return_node())

    # TODO(gty): refactor register engine attribute as a common process
    ret_eng = try_cvt_to_engine_func(module, subgraph, group_name, grp_calib_data)
    if (ret_eng is None):
        return
    attr_name, eng_type = ret_eng

    if adapt_number_ios:
        # NB: Only do number types adaption after the subgraph conversion success
        _adapt_node_number_outputs(module.forward.graph, node)
        _adapt_node_number_inputs(module.forward.graph, node)

    graph = module.forward.graph
    engine_holder = graph.input_list()[0]

    replace_group_with_engine(graph, engine_holder, node, attr_name, eng_type)


def group_nodes(block):
    grp_nodes = []
    for node in block.node_list():
        if (node.kind() == 'prim::FusionGroup'):
            grp_nodes.append(node)
        for blk in node.blocks():
            grp_nodes += group_nodes(blk)
    return grp_nodes


def group_to_engine_conversion(
        module,
        try_cvt_to_engine_func,
        adapt_number_ios=False,
        quantization_calib_file=None
):
    if (isinstance(module, torch.jit.ScriptModule)):
        module = module._c
    assert(isinstance(module, torch._C.ScriptModule))

    graph = module.forward.graph

    if (not utils.graph_in_topology_order(graph)):
        raise RuntimeError(
            "Graph nodes not in topology order, please report a bug")

    # in case that there exists trt engines in the graph
    # exist_engines = torch_blade.tensorrt.collect_engines(module)
    exist_engines = False
    if exist_engines:
        engine_names = [i[0] for i in exist_engines]
        # may be get id in a unified place
        engine_id = [int(i.lstrip(tensorrt._TRT_GROUP_NAME).split('_')[0]) for i in engine_names]
        start_id = sorted(engine_id)[-1] + 1
    else:
        start_id = 0
    success_grp_num = 0

    fusion_group_nodes = group_nodes(graph)
    all_calib_data = None
    if quantization_calib_file is not None:
        all_calib_data = torch.load(quantization_calib_file)
        if len(all_calib_data) != len(fusion_group_nodes):
            logger.warning("The number of quantization calibration data "
                           "is not equal to the number of fusion group nodes")
            all_calib_data = None

    for idx, node in enumerate(group_nodes(graph)):
        group_id = success_grp_num + start_id
        grp_calib_data = all_calib_data[idx] if all_calib_data is not None else None
        group_node_to_engine(
            module,
            node,
            try_cvt_to_engine_func,
            adapt_number_ios,
            (group_id, idx),
            grp_calib_data
        )
        success_grp_num += 1

    utils.block_topology_ajust(graph)
    if (not utils.graph_in_topology_order(graph)):
        raise RuntimeError(
            "Graph nodes not in topology order, please report a bug")

    for node in group_nodes(graph):
        if (node.kind() != 'prim::FusionGroup'):
            continue
        subgraph = node.g('Subgraph')
        _inline_node_with_subgraph(graph, node, subgraph)

    # TODO: something like aten::detach will not be constfold but should be
    pass_manager._jit_pass_dce_during_lower_to_trt(module.forward.graph)
    pass_manager._jit_pass_lint(module.forward.graph)

    # TODO: Force Method GraphExecutor update by copy the torch._C.ScriptModule, and
    # return a the new module. The Method maintains a GraphExecutor which is used to
    # actually execute the Graph that defines the method. Currently the GraphExecutor
    # is created and optimized at first time the method is called. After that the graph
    # should not change.

# Copyright 2023 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from onnx import helper, numpy_helper
from onnx.onnx_ml_pb2 import ModelProto

from torch_quant.mnn.onnx_utils import (
    get_constants,
    get_inp_to_node,
    get_node_attr_by_name,
    get_out_to_node,
    is_per_channel_fake_quant,
    is_fake_quant,
    remove_node,
)


def _onnx_pass_remove_identity(model: ModelProto) -> ModelProto:
    graph = model.graph
    constants = get_constants(graph)
    inp_to_node = get_inp_to_node(graph)
    identity_to_be_removed = []
    for node in graph.node:
        if node.op_type != 'Identity':
            continue
        if len(node.input) != 1 or len(node.output) != 1:
            # make the conditions more restrictive
            continue
        input_name = node.input[0]
        output_name = node.output[0]
        if input_name in constants and output_name in inp_to_node:
            for inp_idx, consumer_node in inp_to_node[output_name]:
                consumer_node.input[inp_idx] = input_name
            identity_to_be_removed.append(node.name)
    remove_node(model, identity_to_be_removed)


def _onnx_pass_remove_transpose(model: ModelProto) -> ModelProto:
    """
    Const -> FakeQuantize -> Transpose
    =>
    transposed Const -> FakeQuantize
    """
    graph = model.graph
    inp_to_node = get_inp_to_node(graph)
    out_to_node = get_out_to_node(graph)
    constants = {init.name: init for init in graph.initializer}
    nodes_to_remove = list()
    transpose_nodes = [node for node in graph.node if node.op_type == 'Transpose']
    for node in transpose_nodes:
        if not is_per_channel_fake_quant(out_to_node[node.input[0]]):
            continue
        fake_quant_node = out_to_node[node.input[0]]
        tensor = constants.get(fake_quant_node.input[0])
        if tensor:
            array = numpy_helper.to_array(tensor)
            perm = get_node_attr_by_name(node, 'perm')
            transposed_array = np.transpose(array, perm)
            transposed_tensor = helper.make_tensor(
                name=tensor.name,
                data_type=tensor.data_type,
                dims=transposed_array.shape,
                vals=transposed_array.tobytes(),
                raw=True,
            )
            graph.initializer.remove(tensor)
            graph.initializer.extend([transposed_tensor])
            for attr in fake_quant_node.attribute:
                if attr.name == 'ch_axis':
                    attr.i = perm[attr.i]
            if node.output[0] in inp_to_node:
                for inp_idx, consumer_node in inp_to_node[node.output[0]]:
                    consumer_node.input[inp_idx] = fake_quant_node.output[0]
            nodes_to_remove.append(node.name)
    remove_node(model, nodes_to_remove)


def _onnx_pass_remove_fake_quant(model: ModelProto) -> ModelProto:
    graph = model.graph
    inp2node = get_inp_to_node(graph)
    fake_quant_nodes = [n for n in graph.node if is_fake_quant(n)]
    for node in fake_quant_nodes:
        input_name = node.input[0]
        output_name = node.output[0]
        if output_name in inp2node:
            for inp_idx, consumer_node in inp2node[output_name]:
                consumer_node.input[inp_idx] = input_name
        for out in graph.output:
            if out.name == output_name:
                out.name = input_name
    nodes_to_remove = [n.name for n in fake_quant_nodes]
    remove_node(model, nodes_to_remove)
    inp2node = get_inp_to_node(model.graph)
    out2node = get_out_to_node(model.graph)
    fq_inputs = [i for node in fake_quant_nodes for i in node.input[1:]]
    nodes_to_remove = [out2node[i].name for i in fq_inputs if i not in inp2node]
    remove_node(model, nodes_to_remove)


def onnx_preprocess(model: ModelProto) -> ModelProto:
    _onnx_pass_remove_identity(model)
    _onnx_pass_remove_transpose(model)
    return model


def onnx_postprocess(model: ModelProto) -> ModelProto:
    _onnx_pass_remove_fake_quant(model)
    return model

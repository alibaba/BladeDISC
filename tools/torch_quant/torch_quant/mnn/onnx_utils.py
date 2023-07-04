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

import os
import tempfile
from typing import Any, Dict, Iterable, List, Tuple, Union

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from onnx.onnx_ml_pb2 import GraphProto, ModelProto, NodeProto

from torch_quant.utils import parse_version, torch_version


def _export_onnx(
    model: torch.nn.Module,
    dummy_input: Union[Tuple[Any, ...], torch.Tensor],
    onnx_path: str,
) -> None:
    if torch_version() < parse_version('1.10.0'):
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            do_constant_folding=True,
            enable_onnx_checker=False,
        )
    else:
        if torch_version() < parse_version('1.11.0'):
            from torch.onnx.utils import ONNXCheckerError

            error_type = ONNXCheckerError
        elif torch_version() < parse_version('1.12.0'):
            from torch.onnx.utils import CheckerError

            error_type = CheckerError
        else:
            from torch.onnx import CheckerError

            error_type = CheckerError
        try:
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                do_constant_folding=True,
            )
        except error_type:
            pass


def export_reload_onnx(
    model: torch.nn.Module, dummy_input: Union[Tuple[Any, ...], torch.Tensor]
) -> ModelProto:
    with tempfile.TemporaryDirectory(prefix='onnx_model') as tmp_dirname:
        onnx_path = os.path.join(tmp_dirname, 'tmp.onnx')
        try:
            _export_onnx(model, dummy_input, onnx_path)
        except Exception:
            raise RuntimeError('Failed to export onnx model.')

        if not os.path.exists(onnx_path):
            raise RuntimeError('Failed to export onnx model to the disk.')

        try:
            onnx_model = onnx.load(onnx_path)
        except Exception:
            raise RuntimeError('Failed to load onnx model from the disk.')

    return onnx_model


def get_out_to_node(onnx_graph: GraphProto) -> Dict[str, NodeProto]:
    return {out: nd for nd in onnx_graph.node for out in nd.output}


def get_inp_to_node(onnx_graph: GraphProto) -> Dict[str, List[Tuple[int, NodeProto]]]:
    inp2node = {}
    for node in onnx_graph.node:
        for inp_idx, inp in enumerate(node.input):
            if inp not in inp2node:
                inp2node[inp] = []
            inp2node[inp].append((inp_idx, node))
    return inp2node


def get_constants(onnx_graph: GraphProto) -> Dict[str, np.ndarray]:
    constants = {}
    for init_const in onnx_graph.initializer:
        name = init_const.name
        constants[name] = numpy_helper.to_array(init_const)
    for node in onnx_graph.node:
        if node.op_type == 'Constant':
            for attr in node.attribute:
                if attr.name == 'value':
                    constants[node.output[0]] = numpy_helper.to_array(attr.t)

    # PyTorch's _jit_pass_onnx_deduplicate_initializers pass
    # will combine all equal parameters into one, and then pass
    # onnx::Identity to generate the output of the corresponding
    # name for downstream use. These outputs should also be treated
    # as constants
    for node in onnx_graph.node:
        if node.op_type == 'Identity':
            input_name = node.input[0]
            if input_name in constants:
                constants[node.output[0]] = constants[input_name]

    return constants


def get_node_attr_by_name(node: NodeProto, name: str) -> Any:
    for attr in node.attribute:
        if attr.name != name:
            continue
        if attr.type == onnx.AttributeProto.AttributeType.INTS:
            return tuple(attr.ints)
        elif attr.type == onnx.AttributeProto.AttributeType.INT:
            return attr.i
        elif attr.type == onnx.AttributeProto.AttributeType.FLOATS:
            return tuple(attr.floats)
        elif attr.type == onnx.AttributeProto.AttributeType.FLOAT:
            return attr.f
        elif attr.type == onnx.AttributeProto.AttributeType.TENSOR:
            return numpy_helper.to_array(attr.t)
        elif attr.type == onnx.AttributeProto.AttributeType.STRING:
            return str(attr.s)
        elif attr.type == onnx.AttributeProto.AttributeType.STRINGS:
            return tuple([str(x) for x in attr.strings])
        else:
            raise Exception(f'Unsupported attribute type: "{attr.type}"')
    raise Exception(f'Attribute "{name}" was not found')


def is_node_relu6(node: NodeProto, onnx_model: ModelProto) -> bool:
    if node.op_type != 'Clip':
        return False

    if len(node.input) == 1:
        clip_min = get_node_attr_by_name(node, 'min')
        clip_max = get_node_attr_by_name(node, 'max')
        return clip_min == 0.0 and clip_max == 6.0

    if len(node.input) == 3:
        clip_min, clip_max = None, None
        min_name, max_name = node.input[1], node.input[2]

        _value = lambda x: numpy_helper.to_array(x).tolist()
        for n in onnx_model.graph.node:
            if n.output[0] == min_name and n.op_type == 'Constant':
                clip_min = _value(n.attribute[0].t)
            if n.output[0] == max_name and n.op_type == 'Constant':
                clip_max = _value(n.attribute[0].t)

        if clip_min is None and clip_max is None:
            for iz in onnx_model.graph.initializer:
                if iz.name == min_name:
                    clip_min = _value(iz)
                if iz.name == max_name:
                    clip_max = _value(iz)
        return clip_min == 0.0 and clip_max == 6.0

    return False


def remove_node(onnx_model: ModelProto, nodes_to_remove: Iterable[str]) -> None:
    graph = onnx_model.graph
    nodes_to_remove = [n for n in graph.node if n.name in nodes_to_remove]
    for node in nodes_to_remove:
        graph.node.remove(node)


def is_per_tensor_fake_quant(node: NodeProto) -> bool:
    return node.op_type == 'FakeQuantizeLearnablePerTensorAffine'


def is_per_channel_fake_quant(node: NodeProto) -> bool:
    return node.op_type == 'FakeQuantizeLearnablePerChannelAffine'


def is_fake_quant(node: NodeProto) -> bool:
    return is_per_tensor_fake_quant(node) or is_per_channel_fake_quant(node)

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

import logging
import uuid
from typing import Any, Dict, Iterable, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
from onnx.onnx_ml_pb2 import ModelProto, NodeProto

from torch_quant.mnn.MNN_compression_pb2 import LayerQuantizeParams, Pipeline
from torch_quant.mnn.onnx_passes import onnx_postprocess, onnx_preprocess
from torch_quant.mnn.onnx_utils import (
    export_reload_onnx,
    get_constants,
    get_inp_to_node,
    get_node_attr_by_name,
    get_out_to_node,
    is_fake_quant,
    is_node_relu6,
    is_per_tensor_fake_quant,
)

LOGGER = logging.getLogger(__name__)

MNN_QUANTIZABLE_OPS_IN_ONNX = ['Conv', 'Gemm', 'MatMul']


class QuantInfo(NamedTuple):
    name: str
    scale: np.ndarray
    zero_point: np.ndarray
    quant_min: int
    quant_max: int


class LayerQuantInfo(NamedTuple):
    weight: QuantInfo
    input: QuantInfo
    output: QuantInfo


class MNNConverter:
    def __init__(self, onnx_model: ModelProto) -> None:
        onnx_model = onnx_preprocess(onnx_model)
        self.onnx_model = onnx_model
        graph = onnx_model.graph
        self.inp2node = get_inp_to_node(graph)
        self.out2node = get_out_to_node(graph)
        self.name2const = get_constants(graph)

    def extract_quant_info(
        self, node: NodeProto, name: Optional[str] = None
    ) -> QuantInfo:
        target_tensor_name, scale_name, zero_point_name = node.input
        quant_info = QuantInfo(
            name=name or target_tensor_name,
            scale=self.name2const[scale_name],
            zero_point=self.name2const[zero_point_name],
            quant_min=get_node_attr_by_name(node, 'quant_min'),
            quant_max=get_node_attr_by_name(node, 'quant_max'),
        )
        return quant_info

    def extract_layer_info(self, node: NodeProto) -> LayerQuantInfo:
        # weight
        weight_quant_node = self.out2node[node.input[1]]
        weight_info = self.extract_quant_info(weight_quant_node)

        # input activation
        input_quant_node = self.out2node[node.input[0]]
        if not is_per_tensor_fake_quant(input_quant_node):
            raise RuntimeError(
                "MNN only supports per-tensor quantization for activation."
            )
        input_info = self.extract_quant_info(input_quant_node)

        # output activation
        output_quant_node, output_name = self.find_output_fake_quant_node(node)
        if output_quant_node is None:
            raise RuntimeError("Cannot find output fake quant node.")
        output_info = self.extract_quant_info(output_quant_node, output_name)

        quant_info = LayerQuantInfo(weight_info, input_info, output_info)

        return quant_info

    def find_output_fake_quant_node(
        self, node: NodeProto
    ) -> Union[Tuple[NodeProto, str], Tuple[None, None]]:
        def _get_output_node(node: NodeProto) -> NodeProto:
            # TODO(wanchen.swc): handle the quantizable nodes with multiple consumers
            output_nodes = self.inp2node[node.output[0]]
            if len(output_nodes) != 1:
                raise NotImplementedError(
                    "Only supports the quantizable nodes with one consumer. "
                    f"Got {len(output_nodes)}."
                )
            return output_nodes[0][1]

        # TODO(wanchen.swc): add relu6 module fusion for mnn, otherwise the
        # corresponding nodes cannot be quantized correctly by mnnconvert.
        def _get_fake_quant_node_with_relu(node: NodeProto) -> Optional[NodeProto]:
            output_node = _get_output_node(node)
            if is_per_tensor_fake_quant(output_node):
                return output_node
            # in onnx opset > 9, F.relu6 is composed by relu + clip
            if is_node_relu6(output_node, self.onnx_model):
                output_node = _get_output_node(output_node)
                if is_per_tensor_fake_quant(output_node):
                    return output_node
            return None

        output_node = _get_output_node(node)

        # 1. QUANTIZABLE_OPS -> fake_quant
        if is_per_tensor_fake_quant(output_node):
            return output_node, output_node.input[0]

        # 2. Conv/Gemm/MatMul -> Relu/Relu6 -> fake_quant
        if node.op_type in ['Conv', 'Gemm', 'MatMul'] and output_node.op_type == 'Relu':
            output_node = _get_fake_quant_node_with_relu(output_node)
            if output_node and node.op_type in ['Conv']:
                return output_node, output_node.input[0]
            if output_node and node.op_type in ['Gemm', 'MatMul']:
                return output_node, node.output[0]

        # 3. MatMul -> Add -> Relu/Relu6 -> fake_quant
        if node.op_type == 'MatMul' and output_node.op_type == 'Add':
            successor_node = _get_output_node(output_node)
            if is_per_tensor_fake_quant(successor_node):
                return successor_node, output_node.output[0]
            if successor_node.op_type == 'Relu':
                successor_node = _get_fake_quant_node_with_relu(successor_node)
                if successor_node:
                    return successor_node, output_node.output[0]

        return None, None

    def quantizable_nodes(self) -> Iterable[NodeProto]:
        for node in self.onnx_model.graph.node:
            if node.op_type not in MNN_QUANTIZABLE_OPS_IN_ONNX:
                continue
            # A node with input & weight being fake quantized would be considered as
            # a quantize_node. Here we do not distinguish between per-channel and
            # per-tensor scheme and let extract_layer_info do this to catch exception.
            act_node_name, w_node_name = node.input[:2]
            if (
                w_node_name not in self.out2node
                or not is_fake_quant(self.out2node[w_node_name])
                or not is_fake_quant(self.out2node[act_node_name])
            ):
                continue
            yield node

    def get_all_quant_info(self) -> Dict[str, LayerQuantInfo]:
        all_quant_info = {}
        for node in self.quantizable_nodes():
            try:
                all_quant_info[node.name] = self.extract_layer_info(node)
            except Exception:
                LOGGER.warning(f"Fail to extract quant info for node: {node.name}")
        return all_quant_info


def extract_quant_info(
    model: torch.nn.Module, dummy_input: Union[Tuple[Any, ...], torch.Tensor]
) -> Tuple[ModelProto, Dict[str, LayerQuantInfo]]:
    onnx_model = export_reload_onnx(model, dummy_input)
    converter = MNNConverter(onnx_model)
    all_quant_info = converter.get_all_quant_info()
    onnx_model = onnx_postprocess(onnx_model)
    return onnx_model, all_quant_info


def convert_mnn_params(all_quant_info: Dict[str, LayerQuantInfo]) -> Pipeline:
    compress_proto = Pipeline()
    compress_proto.version = '0.0.0'
    if compress_proto.mnn_uuid == '':
        compress_proto.mnn_uuid = str(uuid.uuid4())

    def _set_params(
        quant_info: QuantInfo, is_weight: bool = False
    ) -> Union[LayerQuantizeParams.ActivationParams, LayerQuantizeParams.WeightParams]:
        if is_weight:
            params = LayerQuantizeParams.WeightParams()
        else:
            params = LayerQuantizeParams.ActivationParams()
            params.zero_point = int(quant_info.zero_point.tolist()[0])
        # If enable OAQ, this should be replaced by bit num on the onnx graph
        params.bits = 8
        params.name = quant_info.name
        params.scales.extend(quant_info.scale.tolist())
        params.clamp_min = int(quant_info.quant_min)
        params.clamp_max = int(quant_info.quant_max)
        return params

    quant_algorithm = compress_proto.algo.add()
    for quant_info in all_quant_info.values():
        layer = quant_algorithm.quant_params.layer.add()
        layer.input.append(_set_params(quant_info.input))
        layer.output.append(_set_params(quant_info.output))
        layer.weight.append(_set_params(quant_info.weight, is_weight=True))

    return compress_proto

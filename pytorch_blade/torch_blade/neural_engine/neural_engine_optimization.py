# Copyright 2022 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import onnx
from intel_extension_for_transformers.backends.neural_engine.compile.onnx_utils import \
    is_supported_onnx_graph
from torch_blade.clustering.support_fusion_group import supported_node_fusion
from torch_blade.onnx_backends import backend_testbed

_NEURAL_ENGINE_GROUP_NAME = "neural_engine_grp"


def is_neural_engine_supported(onnx_proto):
    onnx_model = onnx.load_from_string(onnx_proto)
    graph = onnx_model.graph
    return is_supported_onnx_graph(graph)


def _get_unsupported_nodes(graph):
    neural_engine_unsupported = backend_testbed.get_unsupported_nodes(
        graph,
        is_neural_engine_supported,
        _NEURAL_ENGINE_GROUP_NAME,
        ignore_device=True,
    )
    return neural_engine_unsupported


def optimize_neural_engine(script_module):
    c_module = script_module._c
    graph = c_module.forward.graph
    neural_engine_unsupported = _get_unsupported_nodes(graph)
    top_level_block = graph
    supported_node_fusion(graph, top_level_block, neural_engine_unsupported)
    return neural_engine_unsupported

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

from io import BytesIO

import onnx
import torch
import torch_blade._torch_blade._backends as _backends

from torch_blade import pass_manager
from torch_blade import utils
from torch_blade.clustering.support_group_conversion import group_to_engine_conversion
from torch_blade.logging import logger


def _deduplicate_onnx_graph_outputs(graph):
    # Note: it is added because, after some onnx graph optimization pass,
    # some output place was optimized to the same variable. However,
    # the duplication in outputs would lead to TensorRT converter failure.
    ret_node = graph.return_node()
    ret_vals = ret_node.input_list()
    ret_node.removeAllInputs()
    visited = set()

    for v in ret_vals:
        if v in visited:
            continue
        ret_node.addInput(v)
        visited.add(v)


def _build_onnx_engine(subgraph, engine_build_func, q_info=None, dynamic_settings=None):
    # pass lower to onnx & constfold
    graph, value_map = pass_manager._jit_pass_lower_to_onnx(subgraph)

    state = _backends.EngineState()
    state.inputs = [_backends.TensorInfo(inp) for inp in graph.inputs()]
    state.outputs = [_backends.TensorInfo(out) for out in graph.outputs()]

    # deduplicate only deduplicate outputs variable's
    # would not invalid value_map
    _deduplicate_onnx_graph_outputs(graph)

    # update q_val for `torchscript -> onnx` if needed
    if q_info is not None:
        q_info = q_info.generate_through_mapping(value_map)

    dynamic_shapes, dynamic_axes = [], None
    if dynamic_settings is not None:
        dynamic_shapes = dynamic_settings
        dynamic_axes = dynamic_shapes[0].dynamic_setting

    dyn_proto = pass_manager._export_onnx(graph, dynamic_axes)
    onnx_model = onnx.load_from_string(dyn_proto)
    if len(onnx_model.graph.node) == 0:
        # input a graph with empty nodes to onnx builder would cause segfault
        return None
    q_val = q_info.q_val if q_info is not None else {}

    state.model_proto = dyn_proto
    return engine_build_func(dyn_proto, state, dynamic_shapes, q_val)

def _subgraph_to_bytes(subgraph, group_name):
    fallback_module = utils.subgraph_to_module(subgraph, group_name)
    m_bytes = BytesIO()
    torch.jit.save(fallback_module, m_bytes)
    return m_bytes.getvalue()


def _register_onnx_engine(
    module, subgraph, state, group_name, disable_fallback=False
):
    if disable_fallback:
        state.model_proto = ""

    fallback_bytes = (
        "" if disable_fallback else _subgraph_to_bytes(subgraph, group_name)
    )
    # register engine into module, something like:
    # __torch__.torch.classes.torch_blade.Engine = prim::GetAttr[name="trt_grp0"](%self)
    eng_type = _backends.register_engine(
        module,
        state,
        group_name,
        fallback_bytes,
        str(subgraph),
    )
    return eng_type


def build_onnx_engine(
    module,
    group_id,
    _try_build_onnx_engine,
    q_info=None,
    disable_fallback=False,
    dynamic_settings=None,
):
    # q_info passed to this function and `try_cvt_to_onnx_func`
    # only contains quantization information for each `prim::FusionGroup`
    def try_cvt_to_onnx_func(c_module, subgraph, group_name, q_info=None):
        # NB: clear all cached memory for onnx tuning
        torch.cuda.empty_cache()
        # NB: some onnx lowering pass would modify the subgraph
        runtime_fallback_subgraph = subgraph.copy()
        subgraph_idx = int(group_name.split("_")[-1])
        grp_dynamic_settings = (
            [s[subgraph_idx] for s in dynamic_settings] if dynamic_settings else None
        )
        try:
            engine_data = _build_onnx_engine(
                subgraph, _try_build_onnx_engine, q_info, grp_dynamic_settings
            )
        except Exception as error:
            logger.warning(error)
            return None

        if engine_data is None:
            return None

        group_name = f"{group_id}{group_name}"
        otype = _register_onnx_engine(
            c_module,
            runtime_fallback_subgraph,
            engine_data,
            group_name,
            disable_fallback,
        )
        return group_name, otype

    group_to_engine_conversion(module, try_cvt_to_onnx_func, q_info)

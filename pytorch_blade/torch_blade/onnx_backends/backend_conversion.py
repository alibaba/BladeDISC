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
import os

import onnx
import torch
import torch_blade._torch_blade._backends as _backends
from torch_blade import pass_manager, tools, utils
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


def _try_cast_graph_integer_inputs_to_i32(graph):
    # Refer to https://github.com/onnx/onnx-tensorrt/blob/main/docs/operators.md.
    # There is limited support for INT32, INT64, and DOUBLE types.
    # Convert INT64 to INT32.
    for val in graph.inputs():
        if val.type().scalarType() == "Long":
            tools.cast_to_i32_tensor_type(val)


def _build_onnx_engine(subgraph, engine_build_func, group_name,
                       dynamic_settings=None, cast_int_to_i32=False, grp_calib_data=None):
    if cast_int_to_i32:
        _try_cast_graph_integer_inputs_to_i32(subgraph)

    # pass lower to onnx & constfold
    graph, value_map = pass_manager._jit_pass_lower_to_onnx(subgraph)

    state = _backends.EngineState()
    state.inputs = [_backends.TensorInfo(inp) for inp in graph.inputs()]
    state.outputs = [_backends.TensorInfo(out) for out in graph.outputs()]
    if grp_calib_data is not None:
        # TODO (bohua): do some type check
        state.calib_data = grp_calib_data

    # deduplicate only deduplicate outputs variable's
    # would not invalid value_map
    _deduplicate_onnx_graph_outputs(graph)

    dynamic_shapes, dynamic_axes = [], None
    if dynamic_settings is not None:
        dynamic_shapes = dynamic_settings
        dynamic_axes = dynamic_shapes[0].dynamic_setting
        min_shape = dynamic_shapes[0].min_shape
        max_shape = dynamic_shapes[0].max_shape
        opt_shapes = dynamic_shapes[0].opt_shapes
        logger.info(
            "Dynamic shape settings, "
            f"min_shape: {str(min_shape)}, "
            f"max_shape: {str(max_shape)}, "
            f"opt_shapes: {str(opt_shapes)}, "
            f"dynamic_axes: {str(dynamic_axes)}."
        )

    dyn_proto = pass_manager._export_onnx(graph, dynamic_axes)
    onnx_model = onnx.load_from_string(dyn_proto)

    if tools.read_bool_from_env('TORCH_BLADE_DEBUG_LOG', False):
        mlir_dump_dir = "dump_dir"
        if not os.path.exists(mlir_dump_dir):
            os.makedirs(mlir_dump_dir)
        onnx_fname = os.path.join(mlir_dump_dir, group_name + ".onnx")
        with open(onnx_fname, 'wb') as f:
            f.write(dyn_proto)

    if len(onnx_model.graph.node) == 0:
        # input a graph with empty nodes to onnx builder would cause segfault
        logger.debug("Skip build engine for onnx model without node.")
        return None

    state.model_proto = dyn_proto
    return engine_build_func(dyn_proto, state, dynamic_shapes)


def _subgraph_to_bytes(subgraph, group_name):
    fallback_module = utils.subgraph_to_module(subgraph, group_name)
    m_bytes = BytesIO()
    torch.jit.save(fallback_module, m_bytes)
    return m_bytes.getvalue()


def _register_onnx_engine(module, subgraph, state, group_name, disable_fallback=False):
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
    disable_fallback=False,
    dynamic_settings=None,
    cast_int_to_i32=False,
    quantization_calib_file=None,
):
    def try_cvt_to_onnx_func(c_module, c_module_lock, subgraph, group_name, grp_calib_data=None):
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
                subgraph,
                _try_build_onnx_engine,
                group_name,
                grp_dynamic_settings,
                cast_int_to_i32,
                grp_calib_data
            )
        except Exception as error:
            logger.warning(f"Building engine exception: {error}")
            return None

        if engine_data is None:
            logger.warning(f"Building engine failed with empty engine binary.")
            return None

        group_name = f"{group_id}{group_name}"
        with c_module_lock:
            otype = _register_onnx_engine(
                c_module,
                runtime_fallback_subgraph,
                engine_data,
                group_name,
                disable_fallback,
            )
        return group_name, otype

    group_to_engine_conversion(
        module,
        try_cvt_to_onnx_func,
        quantization_calib_file=quantization_calib_file
    )

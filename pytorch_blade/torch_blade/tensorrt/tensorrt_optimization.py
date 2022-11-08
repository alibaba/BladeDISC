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

import torch
import torch_blade._torch_blade._tensorrt as _trt
import torch_blade.onnx_backends.onnx_symbolic_opset9_patches  # noqa
from torch_blade.clustering.support_fusion_group import supported_node_fusion
from torch_blade.config import Config
from torch_blade.onnx_backends import backend_testbed
from torch_blade.onnx_backends.backend_conversion import build_onnx_engine
from torch_blade.quantization.prepare_data import get_calib_file_for_each_group
from torch_blade.tensorrt import flags
from torch_blade.tensorrt.dynamic_shapes import get_dynamic_settings

_TRT_GROUP_NAME = "trt_grp"


def _try_build_trt_engine(dyn_proto, state, dynamic_settings, *args, **kwargs):
    # try to convert to tensorrt engine
    cfg = Config.get_current_context_or_new()
    if cfg.enable_int8:
        with flags.builder_flags_context(
                1 << int(flags.BuilderFlag.INT8) | 1 << int(flags.BuilderFlag.FP16)
        ):
            return _trt.cvt_onnx_to_tensorrt(dyn_proto, state, dynamic_settings)
    elif cfg.enable_fp16:
        with flags.builder_flags_context(1 << int(flags.BuilderFlag.FP16)):
            return _trt.cvt_onnx_to_tensorrt(dyn_proto, state, dynamic_settings)
    else:
        return _trt.cvt_onnx_to_tensorrt(dyn_proto, state, dynamic_settings)


def get_unsupported_nodes(graph, ignore_device=False):
    trt_unsupported = backend_testbed.get_unsupported_nodes(
        graph,
        _trt.is_onnx2trt_supported,
        _trt.backend_name(),
        ignore_device=ignore_device,
    )
    return trt_unsupported


def trt_engine_conversion(
        c_module,
        disable_fallback=False,
        dynamic_settings=None,
        quantization_calib_file=None
):
    build_onnx_engine(
        c_module,
        _TRT_GROUP_NAME,
        _try_build_trt_engine,
        disable_fallback,
        dynamic_settings,
        cast_int_to_i32=True,
        quantization_calib_file=quantization_calib_file
    )


def optimize_trt(script_module, disable_fallback=False):
    """
    Given a ScriptModule, do TensorRT optimization on it.

    Args:
        script_module(torch.jit.ScriptModule): PyTorch ScriptModule to be optimized by trt.
    """
    assert isinstance(
        script_module, torch.jit.ScriptModule
    ), "Only torch.jit.ScriptModule can be optimized by TensorRT, but {} is given.".format(
        type(script_module)
    )
    # do tensorrt optimization
    c_module = script_module._c
    graph = c_module.forward.graph
    trt_unsupported = get_unsupported_nodes(graph)
    dynamic_settings = get_dynamic_settings(c_module, trt_unsupported)
    top_level_block = graph
    supported_node_fusion(graph, top_level_block, trt_unsupported)
    with get_calib_file_for_each_group(c_module) as quantization_calib_file:
        trt_engine_conversion(
            c_module, disable_fallback, dynamic_settings, quantization_calib_file
        )

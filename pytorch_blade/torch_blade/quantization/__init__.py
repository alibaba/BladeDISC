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

from torch_blade.config import Config
from torch_blade.mlir import _DISC_NAME
from torch_blade.tensorrt import _TRT_NAME

try:
    import torch_blade._torch_blade._quantization as _quantization
    _is_available = True

except ImportError:
    _is_available = False


def is_available():
    return _is_available


def _jit_pass_add_placeholder_for_fake_quant(c_module):
    _quantization.add_placeholder_for_fake_quant(c_module)


def _jit_pass_remove_all_placeholder(c_module):
    _quantization.remove_placeholder(c_module)


def _jit_replace_aten_fake_quant_with_custom_version(c_module):
    _quantization.replace_aten_fake_quant_with_custom_version(c_module)


def _process_aten_fake_quant(c_module):
    # This function will processes aten::fake_quant to avoid it being folded by
    # constant-folding pass. The process is different under different backend settings.
    # For TensorRT backend, we want to reuse the onnx converter for aten::fake_quant,
    # so a placeholder custom op is added after each aten::fake_quant of the weight.
    # And the placeholder op will be removed after the whole torchscript preprocess passes
    # are done.
    # For DISC backend, aten::fake_quant will be replaced with our custom fake_quant op.
    # And all quantization info needed by DISC will be stored in it.
    cfg = Config.get_current_context_or_new()
    backend = cfg.optimization_pipeline
    if backend == _TRT_NAME:
        # Add placeholder for each aten fake quant node of weight.
        # Or it will be folded by _jit_pass_constant_propagation.
        # TODO: remove this when fake_quant is added to the skip_list
        # of _jit_pass_constant_propagation.
        # https://github.com/pytorch/pytorch/issues/81460
        _jit_pass_add_placeholder_for_fake_quant(c_module)
    elif backend == _DISC_NAME:
        _jit_replace_aten_fake_quant_with_custom_version(c_module)
        # to avoid torch_blade::fake_quant be folded
        _jit_pass_add_placeholder_for_fake_quant(c_module)
    else:
        raise RuntimeError("Unsupported backend for torchscript with fake quant")


def _jit_pass_quantization_preprocess(c_module):
    if not _is_available:
        return

    cfg = Config.get_current_context_or_new()
    is_enabled_quantization = cfg.enable_int8
    if is_enabled_quantization:
        _process_aten_fake_quant(c_module)


def _jit_pass_quantization_postprocess(c_module):
    if not _is_available:
        return

    cfg = Config.get_current_context_or_new()
    is_enabled_quantization = cfg.enable_int8
    if _is_available and is_enabled_quantization:
        _jit_pass_remove_all_placeholder(c_module)


def is_fake_quant_op(inp_node_kind):
    if not _is_available:
        # If quantization is not available, there should not be
        # fake quant node in the torchscript, because the torchscript
        # is generated by blade_compression. I just copy the name of
        # aten::fake_quantize here to make this util function work for
        # other situation.
        fake_quant_name = [
            "aten::fake_quantize_per_tensor_affine",
            "aten::fake_quantize_per_channel_affine",
            "torch_blade::fake_quant"
        ]
    else:
        fake_quant_name = [
            _quantization.at_fake_quant_per_tensor_affine_name,
            _quantization.at_fake_quant_per_channel_affine_name,
            _quantization.torch_blade_fake_quant_name
        ]

    return inp_node_kind in fake_quant_name


def get_fake_quant_node(graph):
    return [n for n in graph.nodes() if is_fake_quant_op(n.kind())]

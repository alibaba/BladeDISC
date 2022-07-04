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
from torch.onnx import symbolic_opset9, symbolic_helper
from torch_blade import utils


def _is_tensor(x):
    return x.type().isSubtypeOf(torch._C.TensorType.get())


def _get_tensor_rank(x):
    if not _is_tensor(x) or x.type() is None:
        return None
    return x.type().dim()


def linear(g, input, weight, bias):
    rank = _get_tensor_rank(input)
    weight = symbolic_opset9.t(g, weight)
    if rank == 2 and not bias.node().mustBeNone():
        alpha = g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64))
        beta = g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64))
        output = symbolic_opset9.addmm(g, bias, input, weight, alpha, beta)
    else:
        output = symbolic_opset9.matmul(g, input, weight)
        if not bias.node().mustBeNone():
            output = symbolic_opset9.add(g, bias, output)

    return output


if not hasattr(symbolic_opset9, "linear"):
    symbolic_opset9.linear = linear

symbolic_opset9_convolution = symbolic_opset9._convolution


def floordiv(g, self, other):
    return symbolic_opset9.floor_divide(g, self, other)


if not hasattr(symbolic_opset9, "floordiv"):
    symbolic_opset9.floordiv = floordiv


# The adapter make parameter allow_tf32 optional for backward compatiblity
@symbolic_helper.parse_args(
    "v", "v", "v", "is", "is", "is", "i", "is", "i", "i", "i", "i", "i"
)
def _convolution(
    g,
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    benchmark,
    deterministic,
    cudnn_enabled,
    allow_tf32=None,
):
    if utils.torch_version_number() >= utils.parse_version("1.7.1"):
        return symbolic_opset9_convolution(
            g,
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            benchmark,
            deterministic,
            cudnn_enabled,
            allow_tf32,
        )
    else:
        return symbolic_opset9_convolution(
            g,
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            benchmark,
            deterministic,
            cudnn_enabled,
        )


symbolic_opset9._convolution = _convolution

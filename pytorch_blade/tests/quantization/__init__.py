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

import unittest

import torch
import torch.nn.functional as F
from torch import nn
from torch_blade.quantization import is_available as is_quantization_available
from torch_blade.testing.common_utils import TestCase
from torch_blade.utils import torch_version_number

TORCH_VERSION = torch_version_number()

if TORCH_VERSION >= (1, 10):
    zero_point_dtype = torch.int32
    zero_point_type_str = "IntType"
else:
    zero_point_dtype = torch.long
    zero_point_type_str = "LongType"


class ModelWithFakeQuant(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3, 4, 3, 3))
        self.weight_scale = nn.Parameter(torch.randn(3))
        self.weight_zero_point = nn.Parameter(torch.zeros(3))
        self.weight_quant_min = -128
        self.weight_quant_max = 127
        self.weight_axis = 0
        self.input_scale = 1.0
        self.input_zero_point = 25
        self.input_quant_min = 0
        self.input_quant_max = 255

    def forward(self, x):
        x = torch.fake_quantize_per_tensor_affine(
            x, scale=self.input_scale, zero_point=self.input_zero_point,
            quant_min=self.input_quant_min, quant_max=self.input_quant_max)
        weight = torch.fake_quantize_per_channel_affine(
            self.weight.data, self.weight_scale.data,
            self.weight_zero_point.data.to(zero_point_dtype),
            axis=self.weight_axis, quant_min=self.weight_quant_min, quant_max=self.weight_quant_max
        )
        y = F.conv2d(x, weight, bias=None)
        return y


class PerTensorFakeQuant(nn.Module):
    def __init__(self, scale, zero_point, quant_min, quant_max):
        super().__init__()
        self.scale = scale
        self.zero_point = zero_point
        self.quant_min = quant_min
        self.quant_max = quant_max

    def forward(self, x):
        x = torch.fake_quantize_per_tensor_affine(
            x, self.scale, self.zero_point,
            self.quant_min, self.quant_max
        )
        return x


class PerChannelFakeQuant(nn.Module):
    def __init__(self, scale, zero_point, quant_min, quant_max, axis):
        super().__init__()
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)
        self.axis = axis
        self.quant_min = quant_min
        self.quant_max = quant_max

    def forward(self, x):
        x = torch.fake_quantize_per_channel_affine(
            x, self.scale, self.zero_point,
            axis=self.axis, quant_min=self.quant_min, quant_max=self.quant_max
        )
        return x


class QuantizationTestCase(TestCase):
    def setUp(self):
        super().setUp()
        self.is_quantization_available = is_quantization_available()
        if not is_quantization_available():
            self.skipTest("Quantization support was not built")

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
from torch_blade.tensorrt import is_available as is_tensorrt_available
from torch_blade.testing.common_utils import TestCase
from torch_blade.utils import torch_version_number

TORCH_VERSION = torch_version_number()

if TORCH_VERSION >= (1, 10):
    zero_point_dtype = torch.int32
else:
    zero_point_dtype = torch.long


def skipIfNoQuantization():
    # only trt backend is supported for quantization
    return unittest.skipIf(
        not is_quantization_available() or not is_tensorrt_available(),
        "Quantization support was not built")


class ModelWithFakeQuant(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3, 4, 3, 3))
        self.weight_scale = nn.Parameter(torch.randn(3))
        self.weight_zero_point = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        x = torch.fake_quantize_per_tensor_affine(x, scale=1.0, zero_point=0, quant_min=-128, quant_max=127)
        weight = torch.fake_quantize_per_channel_affine(
            self.weight.data, self.weight_scale.data,
            self.weight_zero_point.data.to(zero_point_dtype),
            axis=0, quant_min=-128, quant_max=127
        )
        y = F.conv2d(x, weight, bias=None)
        return y


class QuantizationTestCase(TestCase):
    def setUp(self):
        super().setUp()
        self.is_quantization_available = is_quantization_available()

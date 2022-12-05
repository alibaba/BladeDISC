# Copyright 2021 The BladeDISC Authors. All rights reserved.
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
import unittest

import cpuinfo
import torch
from tests.disc.testing_base import CPUDiscPdlQuantizationTestCase
from tests.quantization import zero_point_dtype
from torch import nn
from torch.nn import functional as F
from torch_blade.utils import torch_version_number

TORCH_VERSION = torch_version_number()


def cpu_support_vnni():
    cpu_info = cpuinfo.get_cpu_info()
    return 'avx512_vnni' in cpu_info['flags']


class CPUDiscPdlQuantizationE2ETestCase(CPUDiscPdlQuantizationTestCase):
    def setUp(self):
        super().setUp()
        if not cpu_support_vnni():
            self.skipTest("Quantization x86 test case only works on cpu"
                          " with vnni")


@unittest.skipIf(TORCH_VERSION < (1, 9),
                 "The patterns corresponding to pytorch before version "
                 "1.9.0 has not yet been implemented ")
class TestCPULiner(CPUDiscPdlQuantizationE2ETestCase):
    def test_s8s8s8s32_per_channel_with_bias(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_scale = 0.1
                self.input_zero_point = 0
                self.output_scale = 0.2
                self.output_zero_point = 0
                self.register_buffer("weight_scale", torch.randn(64))
                self.register_buffer("weight_zero_point",
                                     torch.zeros(64).to(zero_point_dtype))
                self.register_buffer("bias_zero_point",
                                     torch.zeros(64).to(zero_point_dtype))
                self.weight_quant_min = -128
                self.weight_quant_max = 127
                self.activation_quant_min = -128
                self.activation_quant_max = 127
                self.bias_quant_min = -2**31
                self.bias_quant_max = 2**31 - 1
                self.register_buffer("weight", torch.randn(64, 64))
                self.register_buffer("bias", torch.randn(64))
                self.ch_axis = 0

            def forward(self, x):
                x = torch.fake_quantize_per_tensor_affine(
                    x, self.input_scale, self.input_zero_point,
                    self.activation_quant_min, self.activation_quant_max
                )
                weight = torch.fake_quantize_per_channel_affine(
                    self.weight, self.weight_scale, self.weight_zero_point,
                    self.ch_axis, self.weight_quant_min, self.weight_quant_max
                )
                if not torch.jit.is_tracing():
                    # Limited by the current architecture, it is not easy to
                    # add a fake-quant to bias in blade_compression. In order
                    # to test the accuracy of the quantized disc model, we
                    # simulated the quantization of the bias during the
                    # forward inference process of nn.Module.
                    bias_scale = self.input_scale * self.weight_scale
                    quant_bias = torch.fake_quantize_per_channel_affine(
                        self.bias, bias_scale, self.bias_zero_point,
                        self.ch_axis, self.bias_quant_min, self.bias_quant_max
                    )
                    bias = quant_bias
                else:
                    bias = self.bias
                x = F.linear(x, weight, bias=bias)
                x = torch.fake_quantize_per_tensor_affine(
                    x, self.output_scale, self.output_zero_point,
                    self.activation_quant_min, self.activation_quant_max
                )
                return x

        pdll_files = [
            os.path.join(self.common_pdll_dir, "fake_quant.pdll"),
            os.path.join(self.device_pdll_dir, "dequant_gemm_quant.pdll")
        ]
        pdll_files = ",".join(pdll_files)
        inp = torch.randn(1, 64).to(self.device)
        model = Model().eval().to(self.device)
        self._test_e2e(model, inp, pdll_files=pdll_files, enable_int8=True)

    def test_s8s8s8_per_channel_without_bias(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_scale = 0.1
                self.input_zero_point = 0
                self.output_scale = 0.2
                self.output_zero_point = 0
                self.register_buffer("weight_scale", torch.randn(64))
                self.register_buffer("weight_zero_point",
                                     torch.zeros(64).to(zero_point_dtype))
                self.register_buffer("bias_zero_point",
                                     torch.zeros(64).to(zero_point_dtype))
                self.weight_quant_min = -128
                self.weight_quant_max = 127
                self.activation_quant_min = -128
                self.activation_quant_max = 127
                self.bias_quant_min = -2**31
                self.bias_quant_max = 2**31 - 1
                self.register_buffer("weight", torch.randn(64, 64))
                self.register_buffer("bias", torch.randn(64))
                self.ch_axis = 0

            def forward(self, x):
                x = torch.fake_quantize_per_tensor_affine(
                    x, self.input_scale, self.input_zero_point,
                    self.activation_quant_min, self.activation_quant_max
                )
                weight = torch.fake_quantize_per_channel_affine(
                    self.weight, self.weight_scale, self.weight_zero_point,
                    self.ch_axis, self.weight_quant_min, self.weight_quant_max
                )
                x = F.linear(x, weight, bias=None)
                x = torch.fake_quantize_per_tensor_affine(
                    x, self.output_scale, self.output_zero_point,
                    self.activation_quant_min, self.activation_quant_max
                )
                return x

        pdll_files = [
            os.path.join(self.common_pdll_dir, "fake_quant.pdll"),
            os.path.join(self.device_pdll_dir, "dequant_gemm_quant.pdll")
        ]
        pdll_files = ",".join(pdll_files)
        inp = torch.randn(1, 64).to(self.device)
        model = Model().eval().to(self.device)
        self._test_e2e(model, inp, pdll_files=pdll_files, enable_int8=True)


if __name__ == "__main__":
    unittest.main()

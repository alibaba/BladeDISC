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

import os
import unittest

import torch
from tests.disc.testing_base import GPUDiscPdlQuantizationTestCase
from torch import nn
from torch.nn import functional as F
from torch_blade.utils import torch_version_number

TORCH_VERSION = torch_version_number()

class GPUDiscPdlQuantizationE2ETestCase(GPUDiscPdlQuantizationTestCase):
    def setUp(self):
        super().setUp()
        support_cuda_version = ['11.3', '11.7']
        if torch.version.cuda not in support_cuda_version:
            self.skipTest("Currently, the correctness can be ensured on cuda"
                          " version 11.3/11.7. Using other versions may cause"
                          " correctness problem ")

@unittest.skipIf(TORCH_VERSION < (1, 9),
                 "The patterns corresponding to pytorch before version "
                 "1.9.0 has not yet been implemented ")
class TestGPUQuantizedDotMerge(GPUDiscPdlQuantizationE2ETestCase):
    def test_qkv_dot_merge(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_scale = 0.1
                self.input_zero_point = 0
                self.output_scale = 0.05
                self.output_zero_point = 0
                self.weight_scale = 0.3
                self.weight_zero_point = 0
                self.weight_quant_min = -128
                self.weight_quant_max = 127
                self.activation_quant_min = -128
                self.activation_quant_max = 127
                self.register_buffer("weight1", torch.randn(512, 512))
                self.register_buffer("bias1", torch.randn(512))
                self.register_buffer("weight2", torch.randn(512, 512))
                self.register_buffer("bias2", torch.randn(512))
                self.register_buffer("weight3", torch.randn(512, 512))
                self.register_buffer("bias3", torch.randn(512))
                self.bias_scale = 0.2
                self.bias_zero_point = 0
                # bias i8 quantization
                self.bias_quant_min = -128
                self.bias_quant_max = 127
                self.ch_axis = 0

            def forward(self, x):
                x = torch.fake_quantize_per_tensor_affine(
                    x, self.input_scale, self.input_zero_point,
                    self.activation_quant_min, self.activation_quant_max
                )
                weight1 = torch.fake_quantize_per_tensor_affine(
                    self.weight1, self.weight_scale, self.weight_zero_point,
                    self.weight_quant_min, self.weight_quant_max
                )
                weight2 = torch.fake_quantize_per_tensor_affine(
                    self.weight2, self.weight_scale, self.weight_zero_point,
                    self.weight_quant_min, self.weight_quant_max
                )
                weight3 = torch.fake_quantize_per_tensor_affine(
                    self.weight3, self.weight_scale, self.weight_zero_point,
                    self.weight_quant_min, self.weight_quant_max
                )
                x1 = F.linear(x, weight1, bias=self.bias1)
                x2 = F.linear(x, weight2, bias=self.bias2)
                x3 = F.linear(x, weight3, bias=self.bias3)
                x1 = torch.fake_quantize_per_tensor_affine(
                    x1, self.output_scale, self.output_zero_point,
                    self.activation_quant_min, self.activation_quant_max
                )
                x2 = torch.fake_quantize_per_tensor_affine(
                    x2, self.output_scale, self.output_zero_point,
                    self.activation_quant_min, self.activation_quant_max
                )
                x3 = torch.fake_quantize_per_tensor_affine(
                    x3, self.output_scale, self.output_zero_point,
                    self.activation_quant_min, self.activation_quant_max
                )
                return x1+x2+x3
        model = Model().eval().to(self.device)
        inp = torch.randn(512, 512).to(self.device)
        traced_model = torch.jit.trace(model, inp)
        # only cuda version 11.3/11.7 can be ensured correctness
        if torch.version.cuda == '11.3':
            qgemm_pdl_file = "dequant_gemm_quant_bias_quant.pdll"
        else:
            qgemm_pdl_file = "dequant_gemm_quant_bias_f32_quant.pdll"
        pdll_files = [
            os.path.join(self.common_pdll_dir, "fake_quant.pdll"),
            os.path.join(self.device_pdll_dir, qgemm_pdl_file)
        ]
        pdll_files = ",".join(pdll_files)
        self._test_e2e(model, inp, pdll_files=pdll_files, enable_int8=True, diff_scale=3*model.output_scale)

if __name__ == "__main__":
    unittest.main()

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

import torch
from tests.disc.testing_base import DiscPdlCase
from tests.quantization import zero_point_dtype
from torch import nn
from torch.nn import functional as F
from torch_blade.utils import torch_version_number

TORCH_VERSION = torch_version_number()


@unittest.skipIf(TORCH_VERSION < (1, 9),
                 "The patterns corresponding to pytorch before version "
                 "1.9.0 has not yet been implemented ")
class TestLinear(DiscPdlCase):
    #  weight -> fake-quant  \
    #  input  -> fake-quant -> linear -> fake-quant -> output
    def test_s8s8s8_per_channel(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_scale = 0.1
                self.input_zero_point = 0
                self.output_scale = 0.2
                self.output_zero_point = 0
                self.register_buffer("weight_scale", torch.randn(128))
                self.register_buffer("weight_zero_point", torch.zeros(128).to(zero_point_dtype))
                self.weight_quant_min = -128
                self.weight_quant_max = 127
                self.activation_quant_min = -128
                self.activation_quant_max = 127
                self.register_buffer("weight", torch.randn(128, 128))
                self.register_buffer("bias", torch.randn(128))
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
                x = F.linear(x, weight, bias=self.bias)
                x = torch.fake_quantize_per_tensor_affine(
                    x, self.output_scale, self.output_zero_point,
                    self.activation_quant_min, self.activation_quant_max
                )
                return x

        model = Model().eval().to(self.device)
        inp = torch.randn(1, 2, 128).to(self.device)
        traced_model = torch.jit.trace(model, inp)
        pdll_files = [
            os.path.join(self.common_pdll_dir, "fake_quant.pdll"),
            os.path.join(self.device_pdll_dir, "dequant_gemm_quant.pdll")
        ]
        pdll_files = ",".join(pdll_files)
        expect_str = """
module {
  func.func @main(%arg1: tensor<1x2x128xf32>) -> tensor<1x2x128xf32> attributes {tf.entry_function = {input_placements = "cpu,cpu", inputs = "self,x.1", output_placements = "cpu", outputs = "40"}} {
    %0 = mhlo.constant dense_resource<__elided__> : tensor<128xf32>
    %1 = mhlo.constant dense<2.000000e-01> : tensor<f32>
    %2 = mhlo.constant dense<0> : tensor<128xi32>
    %3 = mhlo.constant dense_resource<__elided__> : tensor<128xf32>
    %4 = mhlo.constant dense_resource<__elided__> : tensor<128x128xf32>
    %5 = mhlo.constant dense<0> : tensor<i32> [unknown]
    %6 = mhlo.constant dense<1.000000e-01> : tensor<f32> [unknown]
    # CHECK: mhlo_disc.quantize
    # CHECK-SAME: axis = dense<>
    # CHECK-SAME: quant_max = 127
    # CHECK-SAME: quant_min = -128
    # CHECK-SAME: use_symmetric = true
    %7 = "mhlo_disc.quantize"(%arg1, %6, %5) {axis = dense<> : tensor<0xi64>, quant_max = 127 : i64, quant_min = -128 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<1x2x128xf32>, tensor<f32>, tensor<i32>) -> tensor<1x2x128xi8>
    # CHECK: mhlo_disc.quantize
    # CHECK-SAME: dense<0>
    # CHECK-SAME: quant_max = 127
    # CHECK-SAME: quant_min = -128
    # CHECK-SAME: use_symmetric = true
    %8 = "mhlo_disc.quantize"(%4, %3, %2) {axis = dense<0> : tensor<1xi64>, quant_max = 127 : i64, quant_min = -128 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<128x128xf32>, tensor<128xf32>, tensor<128xi32>) -> tensor<128x128xi8>
    # CHECK-NOT: mhlo_disc.dequantize
    # CHECK: mhlo_disc.custom_call_v2
    # CHECK-SAME: call_target_name = "disc.custom_call.ral_qgemm"
    # CHECK-SAME: custom_attrs = {}
    %9 = "mhlo_disc.custom_call_v2"(%7, %8, %0, %6, %5, %3, %2, %1, %5) {call_target_name = "disc.custom_call.ral_qgemm", custom_attrs = {}, device = "h", expected_input_layouts = "*,*,*,*,*,*,*,*,*", expected_output_layouts = "*", has_side_effect = false, input_layouts = "*,*,*,*,*,*,*,*,*", input_placements = "h,h", output_layouts = "*", output_placements = "h"} : (tensor<1x2x128xi8>, tensor<128x128xi8>, tensor<128xf32>, tensor<f32>, tensor<i32>, tensor<128xf32>, tensor<128xi32>, tensor<f32>, tensor<i32>) -> tensor<1x2x128xi8>
    # CHECK-NOT: mhlo_disc.quantize
    # CHECK: mhlo_disc.dequantize
    # CHECK-SAME: use_symmetric = true
    %10 = "mhlo_disc.dequantize"(%9, %1, %5) {axis = dense<> : tensor<0xi64>, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<1x2x128xi8>, tensor<f32>, tensor<i32>) -> tensor<1x2x128xf32> [unknown]
    return %10 : tensor<1x2x128xf32>
  }
}
        """
        self._test_torchscipte_to_mhlo(traced_model._c, expect_str, pdll_files, enable_int8=True)


if __name__ == "__main__":
    unittest.main()

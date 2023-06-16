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
from tests.disc.testing_base import CPUDiscPdlQuantizationTestCase, GPUDiscPdlQuantizationTestCase
from tests.quantization import zero_point_dtype
from torch import nn
from torch.nn import functional as F
from torch_blade.utils import torch_version_number

TORCH_VERSION = torch_version_number()


@unittest.skipIf(TORCH_VERSION < (1, 9),
                 "The patterns corresponding to pytorch before version "
                 "1.9.0 has not yet been implemented ")
class TestLinear(CPUDiscPdlQuantizationTestCase):
    #  weight -> fake-quant  \
    #  input  -> fake-quant -> linear -> fake-quant -> output
    def test_s8s8s8f32_per_channel(self):
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
  func.func @main(%arg1: tensor<1x2x128xf32>) -> tensor<1x2x128xf32> attributes {tf.entry_function = {inputs = "x.1", output_placements = "cpu", outputs = "40"}} {
    # CHECK: tensor<128x128xi8>
    %0 = mhlo.constant dense_resource<__elided__> : tensor<128x128xi8>
    %1 = mhlo.constant dense<2.000000e-01> : tensor<f32>
    %2 = mhlo.constant dense<0> : tensor<i32>
    %3 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %4 = mhlo.constant dense_resource<__elided__> : tensor<128xf32>
    %5 = mhlo.constant dense<0> : tensor<128xi32>
    %6 = mhlo.constant dense_resource<__elided__> : tensor<128xf32>
    # CHECK: mhlo_disc.quantize
    # CHECK-SAME: axis = dense<>
    # CHECK-SAME: quant_max = 127
    # CHECK-SAME: quant_min = -128
    # CHECK-SAME: use_symmetric = true
    %7 = "mhlo_disc.quantize"(%arg1, %3, %2) {axis = dense<> : tensor<0xi64>, quant_max = 127 : i64, quant_min = -128 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<1x2x128xf32>, tensor<f32>, tensor<i32>) -> tensor<1x2x128xi8>
    # CHECK-NOT: mhlo_disc.dequantize
    # CHECK: mhlo_disc.custom_call_v2
    # CHECK-SAME: call_target_name = "ral_pdll_qgemm"
    # CHECK-SAME: custom_attrs = {bias_is_const = true, transpose_a = false, transpose_b = false, weight_is_const = true}
    %9 = "mhlo_disc.custom_call_v2"(%7, %0, %6, %3, %2, %4, %5, %1, %2) {call_target_name = "ral_pdll_qgemm", custom_attrs = {bias_is_const = true, transpose_a = false, transpose_b = false, weight_is_const = true}, device = "h", expected_input_layouts = "*,*,*,*,*,*,*,*,*", expected_output_layouts = "*", has_side_effect = false, input_layouts = "*,*,*,*,*,*,*,*,*", input_placements = "h,h", output_layouts = "*", output_placements = "h"} : (tensor<1x2x128xi8>, tensor<128x128xi8>, tensor<128xf32>, tensor<f32>, tensor<i32>, tensor<128xf32>, tensor<128xi32>, tensor<f32>, tensor<i32>) -> tensor<1x2x128xi8>
    # CHECK-NOT: mhlo_disc.quantize
    # CHECK: mhlo_disc.dequantize
    # CHECK-SAME: use_symmetric = true
    %10 = "mhlo_disc.dequantize"(%9, %1, %5) {axis = dense<> : tensor<0xi64>, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<1x2x128xi8>, tensor<f32>, tensor<i32>) -> tensor<1x2x128xf32> [unknown]
    return %10 : tensor<1x2x128xf32>
  }
}
        """
        self._test_torchscipte_to_mhlo(traced_model._c, expect_str, pdll_files, enable_int8=True)

    def test_s8s8s8s32_per_tensor(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_scale = 0.1
                self.input_zero_point = 0
                self.output_scale = 0.2
                self.output_zero_point = 0
                self.weight_scale = 0.3
                self.weight_zero_point = 0
                self.weight_quant_min = -128
                self.weight_quant_max = 127
                self.activation_quant_min = -128
                self.activation_quant_max = 127
                self.register_buffer("weight", torch.randn(128, 128))
                self.register_buffer("bias", torch.randn(128))
                self.bias_scale = self.input_scale * self.weight_scale

            def forward(self, x):
                x = torch.fake_quantize_per_tensor_affine(
                    x, self.input_scale, self.input_zero_point,
                    self.activation_quant_min, self.activation_quant_max
                )
                weight = torch.fake_quantize_per_tensor_affine(
                    self.weight, self.weight_scale, self.weight_zero_point,
                    self.weight_quant_min, self.weight_quant_max
                )
                bias = torch.fake_quantize_per_tensor_affine(
                    self.bias, self.bias_scale, self.weight_zero_point,
                    -2 ** 31, 2 ** 31 - 1
                )
                x = F.linear(x, weight, bias=bias)
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
  func.func @main(%arg1: tensor<1x2x128xf32>) -> tensor<1x2x128xf32> attributes {tf.entry_function = {inputs = "x.1", output_placements = "cpu", outputs = "40"}} {
    # CHECK: tensor<128xi32>
    # CHECK: tensor<128x128xi8>
    %0 = mhlo.constant dense_resource<__elided__> : tensor<128xi32>
    %1 = mhlo.constant dense_resource<__elided__> : tensor<128x128xi8>
    %2 = mhlo.constant dense<3.000000e-01> : tensor<f32>
    %3 = mhlo.constant dense<2.000000e-01> : tensor<f32>
    %4 = mhlo.constant dense<0> : tensor<i32>
    %5 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    # CHECK: mhlo_disc.quantize
    # CHECK-SAME: axis = dense<>
    # CHECK-SAME: quant_max = 127
    # CHECK-SAME: quant_min = -128
    # CHECK-SAME: use_symmetric = true
    %6 = "mhlo_disc.quantize"(%arg1, %5, %4) {axis = dense<> : tensor<0xi64>, quant_max = 127 : i64, quant_min = -128 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<1x2x128xf32>, tensor<f32>, tensor<i32>) -> tensor<1x2x128xi8>    # CHECK-NOT: mhlo_disc.dequantize
    # CHECK-NOT: mhlo_disc.quantize
    # CHECK: mhlo_disc.custom_call_v2
    # CHECK-SAME: call_target_name = "ral_pdll_qgemm"
    # CHECK-SAME: custom_attrs = {bias_is_const = true, transpose_a = false, transpose_b = false, weight_is_const = true}
    %7 = "mhlo_disc.custom_call_v2"(%6, %1, %0, %5, %4, %2, %4, %3, %4) {call_target_name = "ral_pdll_qgemm", custom_attrs = {bias_is_const = true, transpose_a = false, transpose_b = false, weight_is_const = true}, device = "h", expected_input_layouts = "*,AB,*,*,*,*,*,*,*", expected_output_layouts = "*", has_side_effect = false, input_layouts = "*,BA,*,*,*,*,*,*,*", input_placements = "h,h,h,h,h,h,h,h,h", output_layouts = "*", output_placements = "h"} : (tensor<1x2x128xi8>, tensor<128x128xi8>, tensor<128xi32>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>) -> tensor<1x2x128xi8>
    # CHECK-NOT: mhlo_disc.quantize
    # CHECK: mhlo_disc.dequantize
    # CHECK-SAME: use_symmetric = true
    %8 = "mhlo_disc.dequantize"(%7, %3, %4) {axis = dense<> : tensor<0xi64>, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<1x2x128xi8>, tensor<f32>, tensor<i32>) -> tensor<1x2x128xf32>
    return %8 : tensor<1x2x128xf32>
  }
}
        """
        self._test_torchscipte_to_mhlo(traced_model._c, expect_str, pdll_files, enable_int8=True)

    def test_s8s8s8_per_tensor(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_scale = 0.1
                self.input_zero_point = 0
                self.output_scale = 0.2
                self.output_zero_point = 0
                self.weight_scale = 0.3
                self.weight_zero_point = 0
                self.weight_quant_min = -128
                self.weight_quant_max = 127
                self.activation_quant_min = -128
                self.activation_quant_max = 127
                self.register_buffer("weight", torch.randn(128, 128))

            def forward(self, x):
                x = torch.fake_quantize_per_tensor_affine(
                    x, self.input_scale, self.input_zero_point,
                    self.activation_quant_min, self.activation_quant_max
                )
                weight = torch.fake_quantize_per_tensor_affine(
                    self.weight, self.weight_scale, self.weight_zero_point,
                    self.weight_quant_min, self.weight_quant_max
                )
                x = F.linear(x, weight, bias=None)
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
  func.func @main(%arg1: tensor<1x2x128xf32>) -> tensor<1x2x128xf32> attributes {tf.entry_function = {inputs = "x.1", output_placements = "cpu", outputs = "40"}} {
    # CHECK: tensor<128x128xi8>
    %0 = mhlo.constant dense_resource<__elided__> : tensor<128x128xi8>
    %1 = mhlo.constant dense<3.000000e-01> : tensor<f32>
    %2 = mhlo.constant dense<2.000000e-01> : tensor<f32>
    %3 = mhlo.constant dense<0> : tensor<i32>
    %4 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    # CHECK: mhlo_disc.quantize
    # CHECK-SAME: axis = dense<>
    # CHECK-SAME: quant_max = 127
    # CHECK-SAME: quant_min = -128
    # CHECK-SAME: use_symmetric = true
    %5 = "mhlo_disc.quantize"(%arg1, %4, %3) {axis = dense<> : tensor<0xi64>, quant_max = 127 : i64, quant_min = -128 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<1x2x128xf32>, tensor<f32>, tensor<i32>) -> tensor<1x2x128xi8>    # CHECK-NOT: mhlo_disc.dequantize
    # CHECK-NOT: mhlo_disc.quantize
    # CHECK: mhlo_disc.custom_call_v2
    # CHECK-SAME: call_target_name = "ral_pdll_qgemm"
    # CHECK-SAME: custom_attrs = {transpose_a = false, transpose_b = false, weight_is_const = true}
    %6 = "mhlo_disc.custom_call_v2"(%5, %0, %4, %3, %1, %3, %2, %3) {call_target_name = "ral_pdll_qgemm", custom_attrs = {transpose_a = false, transpose_b = false, weight_is_const = true}, device = "h", expected_input_layouts = "*,AB,*,*,*,*,*,*", expected_output_layouts = "*", has_side_effect = false, input_layouts = "*,BA,*,*,*,*,*,*", input_placements = "h,h,h,h,h,h,h,h", output_layouts = "*", output_placements = "h"} : (tensor<1x2x128xi8>, tensor<128x128xi8>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>) -> tensor<1x2x128xi8>     # CHECK-NOT: mhlo_disc.quantize
    # CHECK: mhlo_disc.dequantize
    # CHECK-SAME: use_symmetric = true
    %7 = "mhlo_disc.dequantize"(%6, %2, %3) {axis = dense<> : tensor<0xi64>, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<1x2x128xi8>, tensor<f32>, tensor<i32>) -> tensor<1x2x128xf32>
    return %7 : tensor<1x2x128xf32>
  }
}
        """
        self._test_torchscipte_to_mhlo(traced_model._c, expect_str, pdll_files, enable_int8=True)

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
                x = F.linear(x, weight, bias=None)
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
  func.func @main(%arg1: tensor<1x2x128xf32>) -> tensor<1x2x128xf32> attributes {tf.entry_function = {inputs = "x.1", output_placements = "cpu", outputs = "40"}} {
    # CHECK: tensor<128x128xi8>
    %0 = mhlo.constant dense_resource<__elided__> : tensor<128x128xi8>
    %1 = mhlo.constant dense<2.000000e-01> : tensor<f32
    %2 = mhlo.constant dense<0> : tensor<i32
    %3 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    %4 = mhlo.constant dense_resource<__elided__> : tensor<128xf32>
    %5 = mhlo.constant dense<0> : tensor<128xi32>
    # CHECK: mhlo_disc.quantize
    # CHECK-SAME: axis = dense<>
    # CHECK-SAME: quant_max = 127
    # CHECK-SAME: quant_min = -128
    # CHECK-SAME: use_symmetric = true
    %6 = "mhlo_disc.quantize"(%arg1, %3, %2) {axis = dense<> : tensor<0xi64>, quant_max = 127 : i64, quant_min = -128 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<1x2x128xf32>, tensor<f32>, tensor<i32>) -> tensor<1x2x128xi8> loc(#loc)
    # CHECK-NOT: mhlo_disc.quantize
    # CHECK-NOT: mhlo_disc.dequantize
    # CHECK: mhlo_disc.custom_call_v2
    # CHECK-SAME: call_target_name = "ral_pdll_qgemm"
    # CHECK-SAME: custom_attrs = {transpose_a = false, transpose_b = false, weight_is_const = true}
    %8 = "mhlo_disc.custom_call_v2"(%6, %0, %3, %2, %4, %5, %1, %2) {call_target_name = "ral_pdll_qgemm", custom_attrs = {transpose_a = false, transpose_b = false}, device = "h", expected_input_layouts = "*,AB,*,*,*,*,*,*", expected_output_layouts = "*", has_side_effect = false, input_layouts = "*,BA,*,*,*,*,*,*", input_placements = "h,h,h,h,h,h,h,h", output_layouts = "*", output_placements = "h"} : (tensor<1x2x128xi8>, tensor<128x128xi8>, tensor<f32>, tensor<i32>, tensor<128xf32>, tensor<128xi32>, tensor<f32>, tensor<i32>) -> tensor<1x2x128xi8> loc(#loc)
    %9 = "mhlo_disc.dequantize"(%8, %0, %1) {axis = dense<> : tensor<0xi64>, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<1x2x128xi8>, tensor<f32>, tensor<i32>) -> tensor<1x2x128xf32> loc(#loc)
    return %9 : tensor<1x2x128xf32> loc(#loc)
  }
}
                """
        self._test_torchscipte_to_mhlo(traced_model._c, expect_str, pdll_files, enable_int8=True)


@unittest.skipIf(TORCH_VERSION < (1, 9),
                 "The patterns corresponding to pytorch before version "
                 "1.9.0 has not yet been implemented ")
class TestGPULinear(GPUDiscPdlQuantizationTestCase):
    #  weight -> fake-quant  \
    #  input  -> fake-quant -> linear -> fake-quant -> output
    #  bias   -> fake-quant  /
    def test_s8s8s8_s8bias_per_tensor(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_scale = 0.1
                self.input_zero_point = 0
                self.output_scale = 0.2
                self.output_zero_point = 0
                self.weight_scale = 0.3
                self.weight_zero_point = 0
                self.weight_quant_min = -128
                self.weight_quant_max = 127
                self.activation_quant_min = -128
                self.activation_quant_max = 127
                self.register_buffer("weight", torch.randn(128, 128))
                self.register_buffer("bias", torch.randn(128))
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
                weight = torch.fake_quantize_per_tensor_affine(
                    self.weight, self.weight_scale, self.weight_zero_point,
                    self.weight_quant_min, self.weight_quant_max
                )
                bias = torch.fake_quantize_per_tensor_affine(
                    self.bias, self.bias_scale, self.bias_zero_point,
                    self.bias_quant_min, self.bias_quant_max
                )
                x = F.linear(x, weight, bias=bias)
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
  func.func @main(%arg0: !torch.nn.Module<"__torch__.___torch_mangle_11.Model"> [unknown], %arg1: tensor<1x2x128xf32> [unknown]) -> tensor<1x2x128xf32> attributes {tf.entry_function = {input_placements = "cpu,gpu", inputs = "self,x.1", output_placements = "gpu", outputs = "51"}} {
    # CHECK: tensor<128xi8>
    # CHECK: tensor<128x128xi8>
    %0 = mhlo.constant dense_resource<__elided__> : tensor<128xi8>
    %1 = mhlo.constant dense_resource<__elided__> : tensor<128x128xi8>
    %2 = mhlo.constant dense<3.000000e-01> : tensor<f32>
    %3 = mhlo.constant dense<0> : tensor<i32>
    %4 = mhlo.constant dense<2.000000e-01> : tensor<f32>
    %5 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    # CHECK: mhlo_disc.quantize
    # CHECK-SAME: axis = dense<>
    # CHECK-SAME: quant_max = 127
    # CHECK-SAME: quant_min = -128
    # CHECK-SAME: use_symmetric = true
    %6 = "mhlo_disc.quantize"(%arg1, %5, %3) {axis = dense<> : tensor<0xi64>, quant_max = 127 : i64, quant_min = -128 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<1x2x128xf32>, tensor<f32>, tensor<i32>) -> tensor<1x2x128xi8>
    # CHECK-NOT: mhlo_disc.quantize
    # CHECK-NOT: mhlo_disc.dequantize
    # CHECK: mhlo_disc.custom_call_v2
    # CHECK-SAME: call_target_name = "ral_pdll_qgemm"
    # CHECK-SAME: custom_attrs = {}
    %9 = "mhlo_disc.custom_call_v2"(%6, %1, %0, %5, %3, %2, %3, %4, %3) {call_target_name = "disc.custom_call.ral_pdll_qgemm", custom_attrs = {}, device = "d", expected_input_layouts = "*,*,*,*,*,*,*,*,*", expected_output_layouts = "*", has_side_effect = false, input_layouts = "*,*,*,*,*,*,*,*,*", input_placements = "d,d,d,h,h,h,h,h,h", output_layouts = "*", output_placements = "d"} : (tensor<1x2x128xi8>, tensor<128x128xi8>, tensor<128xi8>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>) -> tensor<1x2x128xi8>
    # CHECK-NOT: mhlo_disc.quantize
    # CHECK: mhlo_disc.dequantize
    # CHECK-SAME: use_symmetric = true
    %10 = "mhlo_disc.dequantize"(%7, %4, %3) {axis = dense<> : tensor<0xi64>, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<1x2x128xi8>, tensor<f32>, tensor<i32>) -> tensor<1x2x128xf32>
    return %10 : tensor<1x2x128xf32>
  }
}
        """
        self._test_torchscipte_to_mhlo(traced_model._c, expect_str, pdll_files, enable_int8=True)

    #  weight -> fake-quant  \
    #  input  -> fake-quant -> linear -> fake-quant -> output
    #                   bias /
    #
    # rewrited graph:
    #
    #  weight -> quant  \
    #  input  -> quant -> qgemm -> dequant -> output
    #  bias   -> quant  /
    def test_s8s8s8_quant_bias_per_tensor(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_scale = 0.1
                self.input_zero_point = 0
                self.output_scale = 0.2
                self.output_zero_point = 0
                self.weight_scale = 0.3
                self.weight_zero_point = 0
                self.weight_quant_min = -128
                self.weight_quant_max = 127
                self.activation_quant_min = -128
                self.activation_quant_max = 127
                self.register_buffer("weight", torch.randn(128, 128))
                self.register_buffer("bias", torch.randn(128))
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
                weight = torch.fake_quantize_per_tensor_affine(
                    self.weight, self.weight_scale, self.weight_zero_point,
                    self.weight_quant_min, self.weight_quant_max
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
            os.path.join(self.device_pdll_dir, "dequant_gemm_quant_bias_quant.pdll")
        ]
        pdll_files = ",".join(pdll_files)
        expect_str = """
module {
  func.func @main(%arg0: !torch.nn.Module<"__torch__.___torch_mangle_11.Model"> [unknown], %arg1: tensor<1x2x128xf32> [unknown]) -> tensor<1x2x128xf32> attributes {tf.entry_function = {input_placements = "cpu,gpu", inputs = "self,x.1", output_placements = "gpu", outputs = "51"}} {
    # CHECK: tensor<128xi8>
    # CHECK: tensor<128x128xi8>
    %0 = mhlo.constant dense_resource<__elided__> : tensor<128xi8>
    %1 = mhlo.constant dense_resource<__elided__> : tensor<128x128xi8>
    %2 = mhlo.constant dense<3.000000e-01> : tensor<f32>
    %3 = mhlo.constant dense<0> : tensor<i32>
    %4 = mhlo.constant dense<2.000000e-01> : tensor<f32>
    %5 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    # CHECK: mhlo_disc.quantize
    # CHECK-SAME: axis = dense<>
    # CHECK-SAME: quant_max = 127
    # CHECK-SAME: quant_min = -128
    # CHECK-SAME: use_symmetric = true
    %6 = "mhlo_disc.quantize"(%arg1, %5, %3) {axis = dense<> : tensor<0xi64>, quant_max = 127 : i64, quant_min = -128 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<1x2x128xf32>, tensor<f32>, tensor<i32>) -> tensor<1x2x128xi8>
    # CHECK-NOT: mhlo_disc.quantize
    # CHECK-NOT: mhlo_disc.dequantize
    # CHECK: mhlo_disc.custom_call_v2
    # CHECK-SAME: call_target_name = "ral_pdll_qgemm"
    # CHECK-SAME: custom_attrs = {}
    %9 = "mhlo_disc.custom_call_v2"(%6, %1, %0, %5, %3, %2, %3, %4, %3) {call_target_name = "disc.custom_call.ral_pdll_qgemm", custom_attrs = {}, device = "d", expected_input_layouts = "*,*,*,*,*,*,*,*,*", expected_output_layouts = "*", has_side_effect = false, input_layouts = "*,*,*,*,*,*,*,*,*", input_placements = "d,d,d,h,h,h,h,h,h", output_layouts = "*", output_placements = "d"} : (tensor<1x2x128xi8>, tensor<128x128xi8>, tensor<128xi8>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>, tensor<f32>, tensor<i32>) -> tensor<1x2x128xi8>
    # CHECK-NOT: mhlo_disc.quantize
    # CHECK: mhlo_disc.dequantize
    # CHECK-SAME: use_symmetric = true
    %10 = "mhlo_disc.dequantize"(%7, %4, %3) {axis = dense<> : tensor<0xi64>, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<1x2x128xi8>, tensor<f32>, tensor<i32>) -> tensor<1x2x128xf32>
    return %10 : tensor<1x2x128xf32>
  }
}
        """
        self._test_torchscipte_to_mhlo(traced_model._c, expect_str, pdll_files, enable_int8=True)

    def test_weight_only_quant_linear_without_bias(self):
        # just test the process of torchscript to torch-mlir
        # there is no limit for k and n
        weight_in_channel = 3
        weight_out_channel = 5
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("weight", torch.randn(weight_out_channel, weight_in_channel))
                self.weight_quant_min = 0
                self.weight_quant_max = 255
                self.register_buffer("weight_scale", torch.randn(weight_out_channel))
                init_zp = 128 * torch.ones(weight_out_channel, dtype=zero_point_dtype)
                self.register_buffer("weight_zero_point", init_zp)
                self.ch_axis = 0

            def forward(self, x):
                weight = torch.fake_quantize_per_channel_affine(
                    self.weight, self.weight_scale, self.weight_zero_point,
                    self.ch_axis, self.weight_quant_min, self.weight_quant_max
                )
                x = F.linear(x, weight, bias=None)
                return x

        model = Model().eval().to(self.device)
        inp = torch.randn(1, 2, 3).to(self.device)
        traced_model = torch.jit.trace(model, inp)
        pdll_files = [
            os.path.join(self.common_pdll_dir, "fake_quant.pdll"),
            os.path.join(self.device_pdll_dir, "weight_only_qgemm.pdll")
        ]
        pdll_files = ",".join(pdll_files)
        expect_str = """
module {
  func.func @main(%arg1: tensor<?x?x?xf32>) -> tensor<?x?x5xf32> {
    # CHECK: tensor<5x3xui8>
    # CHECK: tensor<5xf32>
    # CHECK-NOT: mhlo.quantize
    # CHECK-NOT: mhlo.dequantize
    %0 = mhlo.constant dense : tensor<5x3xui8>
    %1 = mhlo.constant dense : tensor<5xf32>
    %2 = mhlo.constant dense<128> : tensor<5xi32>
    # CHECK: mhlo.transpose
    # CHECK-SAME: permutation = dense<[1, 0]>
    # CHECK-SAME: (tensor<5x3xui8>) -> tensor<3x5xui8>
    %3 = "mhlo.transpose"(%0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<5x3xui8>) -> tensor<3x5xui8>
    # CHECK: mhlo_disc.custom_call_v2
    # CHECK-SAME: call_target_name = "ral_pdll_weight_only_qgemm"
    # CHECK-SAME: custom_attrs = {weight_should_be_reordered = true}
    %4 = "mhlo_disc.custom_call_v2"(%arg1, %3, %1, %2) {call_target_name = "ral_pdll_weight_only_qgemm", custom_attrs = {weight_should_be_reordered = true}, device = "d", expected_input_layouts = "*,*,*,*", expected_output_layouts = "*", has_side_effect = false, input_layouts = "*,*,*,*", input_placements = "d,d,d,d", output_layouts = "*", output_placements = "d"} : (tensor<?x?x?xf32>, tensor<3x5xui8>, tensor<5xf32>, tensor<5xi32>) -> tensor<?x?x5xf32>
    return %4 : tensor<?x?x5xf32>
  } loc(#loc)
} loc(#loc)
#loc1 = loc("/workspace/codes/BladeDISC/pytorch_blade/tests/disc/pdl/test_torchscipt_to_mhlo/test_quantization.py":524:0)


"""
        self._test_torchscipte_to_mhlo(traced_model._c, expect_str, pdll_files, enable_int8=True)

    def test_weight_only_quant_linear_with_bias(self):
        # just test the process of torchscript to torch-mlir
        # there is no limit for k and n
        weight_in_channel = 3
        weight_out_channel = 5
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("weight", torch.randn(weight_out_channel, weight_in_channel))
                self.register_buffer("bias", torch.randn(weight_out_channel))
                self.weight_quant_min = 0
                self.weight_quant_max = 255
                self.register_buffer("weight_scale", torch.randn(weight_out_channel))
                init_zp = 128 * torch.ones(weight_out_channel, dtype=zero_point_dtype)
                self.register_buffer("weight_zero_point", init_zp)
                self.ch_axis = 0

            def forward(self, x):
                weight = torch.fake_quantize_per_channel_affine(
                    self.weight, self.weight_scale, self.weight_zero_point,
                    self.ch_axis, self.weight_quant_min, self.weight_quant_max
                )
                x = F.linear(x, weight, bias=self.bias)
                return x

        model = Model().eval().to(self.device)
        inp = torch.randn(1, 2, 3).to(self.device)
        traced_model = torch.jit.trace(model, inp)
        pdll_files = [
            os.path.join(self.common_pdll_dir, "fake_quant.pdll"),
            os.path.join(self.device_pdll_dir, "weight_only_qgemm.pdll")
        ]
        pdll_files = ",".join(pdll_files)
        expect_str = """
module {
  func.func @main(%arg1: tensor<?x?x?xf32>) -> tensor<?x?x5xf32> {
    # CHECK: tensor<5x3xui8>
    # CHECK: tensor<5xf32>
    # CHECK: tensor<5xf32>
    # CHECK-NOT: mhlo.quantize
    # CHECK-NOT: mhlo.dequantize
    %0 = mhlo.constant dense : tensor<5x3xui8>
    %1 = mhlo.constant dense : tensor<5xf32>
    %2 = mhlo.constant dense<128> : tensor<5xi32>
    %3 = mhlo.constant dense : tensor<5xf32>
    # CHECK: mhlo.transpose
    # CHECK-SAME: permutation = dense<[1, 0]>
    # CHECK-SAME: (tensor<5x3xui8>) -> tensor<3x5xui8>
    %4 = "mhlo.transpose"(%0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<5x3xui8>) -> tensor<3x5xui8>
    # CHECK: mhlo_disc.custom_call_v2
    # CHECK-SAME: call_target_name = "ral_pdll_weight_only_qgemm"
    # CHECK-SAME: custom_attrs = {weight_should_be_reordered = true}
    %5 = "mhlo_disc.custom_call_v2"(%arg1, %4, %3, %1, %2) {call_target_name = "ral_pdll_weight_only_qgemm", custom_attrs = {weight_should_be_reordered = true}, device = "d", expected_input_layouts = "*,*,*,*,*", expected_output_layouts = "*", has_side_effect = false, input_layouts = "*,*,*,*,*", input_placements = "d,d,d,d,d", output_layouts = "*", output_placements = "d"} : (tensor<?x?x?xf32>, tensor<3x5xui8>, tensor<5xf32>, tensor<5xf32>, tensor<5xi32>) -> tensor<?x?x5xf32>
    # CHECK-NOT: torch.aten.add.Tensor
    return %5 : tensor<?x?x5xf32>
  }
}
"""
        self._test_torchscipte_to_mhlo(traced_model._c, expect_str, pdll_files, enable_int8=True)

    def test_weight_only_qlinear_with_add(self):
        # just test the process of torchscript to torch-mlir
        # there is no limit for k and n
        weight_in_channel = 3
        weight_out_channel = 5

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("weight", torch.randn(weight_out_channel, weight_in_channel))
                self.weight_quant_min = 0
                self.weight_quant_max = 255
                self.register_buffer("weight_scale", torch.randn(weight_out_channel))
                init_zp = 128 * torch.ones(weight_out_channel, dtype=zero_point_dtype)
                self.register_buffer("weight_zero_point", init_zp)
                self.ch_axis = 0

            def forward(self, x, y):
                weight = torch.fake_quantize_per_channel_affine(
                    self.weight, self.weight_scale, self.weight_zero_point,
                    self.ch_axis, self.weight_quant_min, self.weight_quant_max
                )
                x = F.linear(x, weight, bias=None)
                x = x + y
                return x

        model = Model().eval().to(self.device)
        x = torch.randn(1, 2, weight_in_channel).to(self.device)
        y = torch.randn(1, 2, weight_out_channel).to(self.device)
        inp = (x, y)
        traced_model = torch.jit.trace(model, inp)
        pdll_files = [
            os.path.join(self.common_pdll_dir, "fake_quant.pdll"),
            os.path.join(self.device_pdll_dir, "weight_only_qgemm.pdll")
        ]
        pdll_files = ",".join(pdll_files)
        expect_str = """
module {
  func.func @main(%arg1: tensor<?x?x?xf32> loc(unknown), %arg2: tensor<?x?x?xf32> -> tensor<?x?x?xf32> {
    # CHECK: tensor<5x3xui8>
    # CHECK:  tensor<5xf32>
    %0 = mhlo.constant dense<[[126, 128, 128], [130, 128, 128], [127, 127, 130], [126, 126, 130], [128, 128, 127]]> : tensor<5x3xui8>
    %1 = mhlo.constant dense<[-0.664962589, -0.394065797, 0.868863224, 0.326392621, -2.07432795]> : tensor<5xf32>
    %2 = mhlo.constant dense<128> : tensor<5xi32>
    %3 = "mhlo.transpose"(%0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<5x3xui8>) -> tensor<3x5xui8>
    %4 = "mhlo_disc.custom_call_v2"(%arg1, %3, %1, %2) {call_target_name = "ral_pdll_weight_only_qgemm", custom_attrs = {weight_should_be_reordered = true}, device = "d", expected_input_layouts = "*,*,*,*", expected_output_layouts = "*", has_side_effect = false, input_layouts = "*,*,*,*", input_placements = "d,d,d,d", output_layouts = "*", output_placements = "d"} : (tensor<?x?x?xf32>, tensor<3x5xui8>, tensor<5xf32>, tensor<5xi32>) -> tensor<?x?x5xf32>
    # CHECK: chlo.broadcast_add
    %5 = chlo.broadcast_add %4, %arg2 : (tensor<?x?x5xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    return %5 : tensor<?x?x?xf32>
  }
}
        """
        self._test_torchscipte_to_mhlo(traced_model._c, expect_str, pdll_files, enable_int8=True)


class TestFakeQuant(CPUDiscPdlQuantizationTestCase):
    def setUp(self):
        super().setUp()
        pdll_files = [
            os.path.join(self.common_pdll_dir, "fake_quant.pdll"),
        ]
        self.pdll_files = ",".join(pdll_files)

    def test_per_tensor_symmetric(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = 0.1
                self.zero_point = 0
                self.quant_min = -128
                self.quant_max = 127

            def forward(self, x):
                x = torch.fake_quantize_per_tensor_affine(
                    x, self.scale, self.zero_point,
                    self.quant_min, self.quant_max
                )
                return x

        model = Model().eval().to(self.device)
        inp = torch.randn(2, 3).to(self.device)
        traced_model = torch.jit.trace(model, inp)
        expect_str = """
module {
  func.func @main(%arg1: tensor<2x3xf32> [unknown]) -> tensor<2x3xf32> attributes {tf.entry_function = {inputs = "x", output_placements = "cpu", outputs = "12"}} {
    %0 = mhlo.constant dense<0> : tensor<i32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    # CHECK: mhlo_disc.quantize
    # CHECK-SAME: axis = dense<>
    # CHECK-SAME: quant_max = 127
    # CHECK-SAME: quant_min = -128
    # CHECK-SAME: round_mode = 1
    # CHECK-SAME: use_dynamic = false
    # CHECK-SAME: use_symmetric = true
    %2 = "mhlo_disc.quantize"(%arg1, %1, %0) {axis = dense<> : tensor<0xi64>, quant_max = 127 : i64, quant_min = -128 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<2x3xf32>, tensor<f32>, tensor<i32>) -> tensor<2x3xi8>
    # CHECK: mhlo_disc.dequantize
    # CHECK-SAME: axis = dense<>
    # CHECK-SAME: round_mode = 1
    # CHECK-SAME: use_dynamic = false
    # CHECK-SAME: use_symmetric = true
    %3 = "mhlo_disc.dequantize"(%2, %1, %0) {axis = dense<> : tensor<0xi64>, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<2x3xi8>, tensor<f32>, tensor<i32>) -> tensor<2x3xf32>
    return %3 : tensor<2x3xf32>
  }
}
        """
        self._test_torchscipte_to_mhlo(traced_model._c, expect_str, self.pdll_files, enable_int8=True)

    def test_per_tensor_affine(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = 0.1
                self.zero_point = 33
                self.quant_min = 0
                self.quant_max = 255

            def forward(self, x):
                x = torch.fake_quantize_per_tensor_affine(
                    x, self.scale, self.zero_point,
                    self.quant_min, self.quant_max
                )
                return x

        model = Model().eval().to(self.device)
        inp = torch.randn(2, 3).to(self.device)
        traced_model = torch.jit.trace(model, inp)
        expect_str = """
module {
  func.func @main(%arg1: tensor<2x3xf32> [unknown]) -> tensor<2x3xf32> attributes {tf.entry_function = {inputs = "x", output_placements = "cpu", outputs = "12"}} {
    %0 = mhlo.constant dense<33> : tensor<i32>
    %1 = mhlo.constant dense<1.000000e-01> : tensor<f32>
    # CHECK: mhlo_disc.quantize
    # CHECK-SAME: axis = dense<>
    # CHECK-SAME: quant_max = 255
    # CHECK-SAME: quant_min = 0
    # CHECK-SAME: round_mode = 1
    # CHECK-SAME: use_dynamic = false
    # CHECK-SAME: use_symmetric = false
    %2 = "mhlo_disc.quantize"(%arg1, %1, %0) {axis = dense<> : tensor<0xi64>, quant_max = 255 : i64, quant_min = 0 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = false} : (tensor<2x3xf32>, tensor<f32>, tensor<i32>) -> tensor<2x3xi8> [unknown]
    # CHECK: mhlo_disc.dequantize
    # CHECK-SAME: axis = dense<>
    # CHECK-SAME: round_mode = 1
    # CHECK-SAME: use_dynamic = false
    # CHECK-SAME: use_symmetric = false
    %3 = "mhlo_disc.dequantize"(%2, %1, %0) {axis = dense<> : tensor<0xi64>, round_mode = 1 : i64, use_dynamic = false, use_symmetric = false} : (tensor<2x3xi8>, tensor<f32>, tensor<i32>) -> tensor<2x3xf32> [unknown]
    return %3 : tensor<2x3xf32>
  }
}
        """
        self._test_torchscipte_to_mhlo(traced_model._c, expect_str, self.pdll_files, enable_int8=True)

    def test_per_channel_symmetric(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("scale", torch.ones(3))
                self.register_buffer("zero_point", torch.zeros(3).to(zero_point_dtype))
                self.quant_min = -128
                self.quant_max = 127
                self.ch_axis = 1

            def forward(self, x):
                x = torch.fake_quantize_per_channel_affine(
                    x, self.scale, self.zero_point, self.ch_axis,
                    self.quant_min, self.quant_max
                )
                return x

        model = Model().eval().to(self.device)
        inp = torch.randn(2, 3).to(self.device)
        traced_model = torch.jit.trace(model, inp)
        expect_str = """
module {
  func.func @main(%arg1: tensor<2x3xf32> [unknown]) -> tensor<2x3xf32> attributes {tf.entry_function = {inputs = "x", output_placements = "cpu", outputs = "12"}} {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<3xf32>
    %1 = mhlo.constant dense<0> : tensor<3xi32>
    # CHECK: mhlo_disc.quantize
    # CHECK-SAME: axis = dense<1>
    # CHECK-SAME: quant_max = 127
    # CHECK-SAME: quant_min = -128
    # CHECK-SAME: round_mode = 1
    # CHECK-SAME: use_dynamic = false
    # CHECK-SAME: use_symmetric = true
    %2 = "mhlo_disc.quantize"(%arg1, %0, %1) {axis = dense<1> : tensor<1xi64>, quant_max = 127 : i64, quant_min = -128 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<2x3xf32>, tensor<3xf32>, tensor<3xi32>) -> tensor<2x3xi8>
    # CHECK: mhlo_disc.dequantize
    # CHECK-SAME: axis = dense<1>
    # CHECK-SAME: round_mode = 1
    # CHECK-SAME: use_dynamic = false
    # CHECK-SAME: use_symmetric = true
    %3 = "mhlo_disc.dequantize"(%2, %0, %1) {axis = dense<1> : tensor<1xi64>, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<2x3xi8>, tensor<3xf32>, tensor<3xi32>) -> tensor<2x3xf32>
    return %3 : tensor<2x3xf32>
  }
}
        """
        self._test_torchscipte_to_mhlo(traced_model._c, expect_str, self.pdll_files, enable_int8=True)

    def test_per_channel_affine(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("scale", torch.ones(3))
                self.register_buffer("zero_point", (33 * torch.ones(3)).to(zero_point_dtype))
                self.quant_min = -128
                self.quant_max = 127
                self.ch_axis = 1

            def forward(self, x):
                x = torch.fake_quantize_per_channel_affine(
                    x, self.scale, self.zero_point, self.ch_axis,
                    self.quant_min, self.quant_max
                )
                return x

        model = Model().eval().to(self.device)
        inp = torch.randn(2, 3).to(self.device)
        traced_model = torch.jit.trace(model, inp)
        expect_str = """
module {
  func.func @main(%arg1: tensor<2x3xf32> [unknown]) -> tensor<2x3xf32> attributes {tf.entry_function = {inputs = "x", output_placements = "cpu", outputs = "12"}} {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<3xf32>
    %1 = mhlo.constant dense<0> : tensor<3xi32>
    # CHECK: mhlo_disc.quantize
    # CHECK-SAME: axis = dense<1>
    # CHECK-SAME: quant_max = 127
    # CHECK-SAME: quant_min = -128
    # CHECK-SAME: round_mode = 1
    # CHECK-SAME: use_dynamic = false
    # CHECK-SAME: use_symmetric = true
    %2 = "mhlo_disc.quantize"(%arg1, %0, %1) {axis = dense<1> : tensor<1xi64>, quant_max = 127 : i64, quant_min = -128 : i64, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<2x3xf32>, tensor<3xf32>, tensor<3xi32>) -> tensor<2x3xi8>
    # CHECK: mhlo_disc.dequantize
    # CHECK-SAME: axis = dense<1>
    # CHECK-SAME: round_mode = 1
    # CHECK-SAME: use_dynamic = false
    # CHECK-SAME: use_symmetric = true
    %3 = "mhlo_disc.dequantize"(%2, %0, %1) {axis = dense<1> : tensor<1xi64>, round_mode = 1 : i64, use_dynamic = false, use_symmetric = true} : (tensor<2x3xi8>, tensor<3xf32>, tensor<3xi32>) -> tensor<2x3xf32>
    return %3 : tensor<2x3xf32>
  }
}
        """
        self._test_torchscipte_to_mhlo(traced_model._c, expect_str, self.pdll_files, enable_int8=True)


if __name__ == "__main__":
    unittest.main()

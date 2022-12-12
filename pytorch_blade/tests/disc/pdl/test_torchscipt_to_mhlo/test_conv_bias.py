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
from tests.disc.testing_base import GPUDiscPdlCase
from tests.quantization import zero_point_dtype
from torch import nn
from torch.nn import functional as F
from torch_blade.utils import torch_version_number

TORCH_VERSION = torch_version_number()

@unittest.skipIf(TORCH_VERSION < (1, 9),
                 "The patterns corresponding to pytorch before version "
                 "1.9.0 has not yet been implemented ")
class TestGPULinear(GPUDiscPdlCase):
    def test_conv2d(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv2d = nn.Conv2d(16, 33, 3, stride=2)

            def forward(self, x):
                return self.conv2d(x)
        model = Model().eval().to(self.device)
        inp = torch.randn(8, 16, 56, 56).to(self.device)
        traced_model = torch.jit.trace(model, inp)
        pdll_files = [
            os.path.join(self.device_pdll_dir, "conv_bias.pdll")
        ]
        pdll_files = ",".join(pdll_files)
        expect_str = """
module {
  func.func @main(%arg0: !torch.nn.Module<"__torch__.___torch_mangle_2.Model"> [unknown], %arg1: tensor<8x16x56x56xf32> [unknown]) -> tensor<?x?x?x?xf32> attributes {tf.entry_function = {input_placements = "cpu,gpu", inputs = "self.1,x", output_placements = "gpu", outputs = "15"}} {
    %0 = mhlo.constant dense_resource<__elided__> : tensor<33x16x3x3xf32> [unknown]
    %1 = mhlo.constant dense_resource<__elided__> : tensor<33xf32> [unknown]
    # CHECK-NOT: torch.aten._convolution
    # CHECK: mhlo_disc.custom_call_v2
    # CHECK-SAME: call_target_name = "ral_pdll_conv_bias"
    # CHECK-SAME: data_format = "NHWC"
    %2 = "mhlo_disc.custom_call_v2"(%arg1, %0, %1) {call_target_name = "ral_pdll_conv_bias", custom_attrs = {data_format = "NHWC", dilation = dense<1> : tensor<2xi64>, groups = 1 : i64, padding = dense<0> : tensor<2xi64>, stride = dense<2> : tensor<2xi64>}, device = "d", expected_input_layouts = "NHWC,OHWI,*", expected_output_layouts = "NHWC", has_side_effect = false, input_layouts = "NCHW,OIHW,*", input_placements = "d,d,d", output_layouts = "NCHW", output_placements = "d"} : (tensor<8x16x56x56xf32>, tensor<33x16x3x3xf32>, tensor<33xf32>) -> tensor<?x?x?x?xf32> /usr/local/lib/python3.8/dist-packages/torch/nn/modules/conv.py:453:0
    return %2 : tensor<?x?x?x?xf32> [unknown]
  } [unknown]
}
        """
        self._test_torchscipte_to_mhlo(traced_model._c, expect_str, pdll_files, enable_int8=False)
    
    def test_conv2d_without_bias(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv2d = nn.Conv2d(16, 33, 3, stride=2, bias=False)

            def forward(self, x):
                return self.conv2d(x)
        model = Model().eval().to(self.device)
        inp = torch.randn(8, 16, 56, 56).to(self.device)
        traced_model = torch.jit.trace(model, inp)
        pdll_files = [
            os.path.join(self.device_pdll_dir, "conv_bias.pdll")
        ]
        pdll_files = ",".join(pdll_files)
        expect_str = """
module {
  func.func @main(%arg0: !torch.nn.Module<"__torch__.___torch_mangle_6.Model"> [unknown], %arg1: tensor<8x16x56x56xf32> [unknown]) -> tensor<?x?x?x?xf32> attributes {tf.entry_function = {input_placements = "cpu,gpu", inputs = "self.1,x", output_placements = "gpu", outputs = "15"}} {
    %0 = mhlo.constant dense_resource<__elided__> : tensor<33x16x3x3xf32> [unknown]
    # CHECK-NOT: mhlo_disc.custom_call_v2
    # CHECK: mhlo.convolution
    %1 = mhlo.convolution(%arg1, %0) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<8x16x56x56xf32>, tensor<33x16x3x3xf32>) -> tensor<?x?x?x?xf32> /usr/local/lib/python3.8/dist-packages/torch/nn/modules/conv.py:453:0
    return %1 : tensor<?x?x?x?xf32> [unknown]
  } [unknown]
} [unknown]
        """
        self._test_torchscipte_to_mhlo(traced_model._c, expect_str, pdll_files, enable_int8=False)


if __name__ == "__main__":
    unittest.main()

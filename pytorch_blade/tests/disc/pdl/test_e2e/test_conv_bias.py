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
from tests.disc.testing_base import GPUDiscPdlCase
from tests.quantization import zero_point_dtype
from torch import nn
from torch.nn import functional as F
from torch_blade import mlir, optimize
from torch_blade.config import Config
from torch_blade.utils import torch_version_number

TORCH_VERSION = torch_version_number()


class GPUDiscPdlConv2dbiasE2ETestCase(GPUDiscPdlCase):
    def setUp(self):
        super().setUp()
    def _test_e2e(
            self, model, inp, enable_int8=False
    ):
        origin_output = model(inp)
        cfg = Config.get_current_context_or_new()
        cfg.optimization_pipeline = mlir.backend_name()
        cfg.enable_int8 = enable_int8
        with cfg:
            opt_model = optimize(model, True, inp)
        now_output = opt_model(inp)
        self.assertTrue(torch.allclose(now_output, origin_output, atol=0.25))


@unittest.skipIf(TORCH_VERSION < (1, 9),
                 "The patterns corresponding to pytorch before version "
                 "1.9.0 has not yet been implemented ")
class TestGPULiner(GPUDiscPdlConv2dbiasE2ETestCase):
    def test_conv2d_bias_verify(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("weight", torch.randn(33, 16, 3, 3))
                self.register_buffer("bias", torch.zeros(33))

            def forward(self, x):
                x = x * 2
                x = x * 4
                return torch.nn.functional.conv2d(x, self.weight, bias=self.bias, stride=2)
        model = Model().eval().to(self.device)
        inp = torch.randn(8, 16, 56, 56).to(self.device)
        self._test_e2e(model, inp, enable_int8=False)
    
    def test_conv2d_wo_bias_verify(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("weight", torch.randn(33, 16, 3, 3))

            def forward(self, x):
                x = x * 2
                x = x * 4
                return torch.nn.functional.conv2d(x, self.weight, stride=2)
        model = Model().eval().to(self.device)
        inp = torch.randn(8, 16, 56, 56).to(self.device)
        
        self._test_e2e(model, inp, enable_int8=False)
    
    def test_conv2d_bias_fp16_verify(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("weight", torch.randn(33, 16, 3, 3))
                self.register_buffer("bias", torch.zeros(33))

            def forward(self, x):
                x = x * 2
                x = x * 4
                return torch.nn.functional.conv2d(x, self.weight, bias=self.bias, stride=2)
        model = Model().eval().to(self.device).half()
        inp = torch.randn(8, 16, 56, 56).to(self.device).half()
        
        self._test_e2e(model, inp, enable_int8=False)

if __name__ == "__main__":
    unittest.main()

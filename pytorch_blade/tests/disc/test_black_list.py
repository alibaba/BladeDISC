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

import torch
import unittest

from torch.testing import FileCheck
from torch_blade.config import Config
from tests.disc.testing_base import DiscTestCase


class TestDiscBlackOps(DiscTestCase):
    def test_black_ops(self):
        @torch.jit.script
        def batchnorm_func(x, running_mean, running_var, weight, bias):
            running_mean = running_mean + bias
            running_mean = running_var * weight
            out_y = torch.nn.functional.batch_norm(
                x, running_mean, running_var, weight, bias
            )
            return out_y + x

        channel = 16
        x = torch.ones([20, channel], device=self.device)
        running_mean = torch.randn(channel, device=self.device)
        running_var = torch.randn(channel, device=self.device)
        weight = torch.randn(channel, device=self.device)
        bias = torch.randn(channel, device=self.device)
        test_data = (x, running_mean, running_var, weight, bias)

        expect_gstr = """graph(%self : __torch__.___torch_mangle_0.gen_func,
      %x.1 : Float(20:16, 16:1, requires_grad=0, device=cuda:0),
      %running_mean : Float(16:1, requires_grad=0, device=cuda:0),
      %running_var.1 : Float(16:1, requires_grad=0, device=cuda:0),
      %weight.1 : Float(16:1, requires_grad=0, device=cuda:0),
      %bias.1 : Float(16:1, requires_grad=0, device=cuda:0)):
  # CHECK-NOT: aten::batch_norm
  %30 : __torch__.torch.classes.torch_blade.InitEngine = prim::GetAttr[name="BladeInit"](%self)
  %31 : None = prim::CallMethod[name="forward"](%30)
  %33 : __torch__.torch.classes.torch_blade.TaoEngine = prim::GetAttr[name="mlir_grp0_len8_0"](%self)
  %34 : Tensor[] = prim::ListConstruct(%x.1, %weight.1, %bias.1, %running_var.1)
  %35 : Tensor[] = prim::CallMethod[name="forward"](%33, %34)
  %37 : Float(20:16, 16:1, requires_grad=0, device=cuda:0) = prim::ListUnpack(%35)
  return (%37)"""
        opt_module = self.cvt_to_disc(batchnorm_func, test_data)
        FileCheck().run(expect_gstr, opt_module.graph)

        expect_gstr = """graph(%self : __torch__.___torch_mangle_0.gen_func,
      %x.1 : Float(20:16, 16:1, requires_grad=0, device=cuda:0),
      %running_mean : Float(16:1, requires_grad=0, device=cuda:0),
      %running_var.1 : Float(16:1, requires_grad=0, device=cuda:0),
      %weight.1 : Float(16:1, requires_grad=0, device=cuda:0),
      %bias.1 : Float(16:1, requires_grad=0, device=cuda:0)):
  # CHECK-COUNT-EXACTLY-1: aten::batch_norm
  %30 : __torch__.torch.classes.torch_blade.InitEngine = prim::GetAttr[name="BladeInit"](%self)
  %31 : None = prim::CallMethod[name="forward"](%30)
  %15 : bool = prim::Constant[value=1]() # /usr/local/lib64/python3.6/site-packages/torch/nn/functional.py:2058:33
  %8 : float = prim::Constant[value=1.0000000000000001e-05]() # :0:0
  %7 : float = prim::Constant[value=0.10000000000000001]() # :0:0
  %6 : bool = prim::Constant[value=0]() # :0:0
  %34 : __torch__.torch.classes.torch_blade.TaoEngine = prim::GetAttr[name="mlir_grp0_len1_0"](%self)
  %35 : Tensor[] = prim::ListConstruct(%running_var.1, %weight.1)
  %36 : Tensor[] = prim::CallMethod[name="forward"](%34, %35)
  %38 : Float(16:1, requires_grad=0, device=cuda:0) = prim::ListUnpack(%36)
  %out_y.1 : Float(20:16, 16:1, requires_grad=0, device=cuda:0) = aten::batch_norm(%x.1, %weight.1, %bias.1, %38, %running_var.1, %6, %7, %8, %15)
  %39 : __torch__.torch.classes.torch_blade.TaoEngine = prim::GetAttr[name="mlir_grp1_len2_1"](%self)
  %40 : Tensor[] = prim::ListConstruct(%out_y.1, %x.1)
  %41 : Tensor[] = prim::CallMethod[name="forward"](%39, %40)
  %43 : Float(20:16, 16:1, requires_grad=0, device=cuda:0) = prim::ListUnpack(%41)
  return (%43)"""
        cfg = Config.get_current_context_or_new()
        cfg.customize_op_black_list = ["aten::batch_norm"]
        with cfg:
            opt_module = self.cvt_to_disc(batchnorm_func, test_data)
            FileCheck().run(expect_gstr, opt_module.graph)


if __name__ == "__main__":
    unittest.main()

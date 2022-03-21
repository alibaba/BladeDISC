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

import unittest
import torch
from torch.testing import FileCheck
from torch_blade import exporter
from torch_blade import pass_manager
from torch_blade import utils
from torch_blade.config import Config
from torch_blade.testing.common_utils import TestCase

class TestPasses(TestCase):
    def test_pass_freeze_rank(self):
        class Model(torch.nn.Module):
            def forward(self, inp):
                if inp.dim() == 0:
                    return 0
                elif inp.dim() == 1:
                    return 1
                elif inp.dim() == 2:
                    return 2
                elif inp.dim() == 3:
                    return 3
                else:
                    return 4

        dummy_input = torch.ones([64, 64])
        model = Model()
        graph = exporter.export(Model(), False, dummy_input).graph
        pass_manager._jit_pass_freeze_rank(graph)

        expect_gstr = """
        graph(%x.1 : Tensor):
          # CHECK-COUNT-EXACTLY-1: prim::Constant[value=2]
          %3 : int = prim::Constant[value=2]()
          return (%3)"""
        FileCheck().run(expect_gstr, graph)

    def test_reorder_raise_exception(self):
        @torch.jit.script
        def has_raise_exception(x):
            x_sum = x.sum()
            if x_sum < 0:
                raise Exception("sum < 0")
            x_numel = x.numel()
            if x_sum < 0:
                raise Exception("x_numel < 0")
            x_mean = x_sum / x.numel()
            if x_mean < 0:
                raise Exception("x_mean < 0")
            return x_mean

        graph = has_raise_exception.graph
        pass_manager._jit_pass_reorder_raise_exception(graph)
        last_3nodes = graph.node_list()[-3:]
        self.assertEqual(len(last_3nodes), 3)
        for node in last_3nodes:
            blks = [blk for blk in node.blocks()]
            self.assertGreater(len(blks), 0)
            blk0_nodes = blks[0].node_list()
            self.assertGreater(len(blk0_nodes), 0)
            n0_kind = blk0_nodes[0].kind()
            self.assertEqual(n0_kind, "prim::RaiseException")


    @unittest.skipIf(utils.torch_version_number() >= utils.parse_version("1.8.1"), 'failed')
    def test_pass_licm(self):
        class Model(torch.nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size
                self.hidden = torch.nn.Linear(self.hidden_size, self.hidden_size)

            def forward(self, x):
                y = x
                for i in range(x.size(0)):
                    y = self.hidden(x)
                return y
        hidden_size = 256
        model = Model(hidden_size)
        x = torch.randn(1, 10, hidden_size)
        module = exporter.export(model, False, x)
        pass_manager._jit_pass_freeze_rank(module.graph)
        expected_y = module.forward(x)
        pass_manager._jit_pass_licm(module.graph)
        y = module.forward(x)
        self.assertEqual(y, expected_y)
        # Check matmul op is before Loop after licm pass
        expect_gstr = """
graph(%self : __torch__.___torch_mangle_0.Model,
      %x.1 : Float(1:2560, 10:256, 256:1)):
  %8 : bool = prim::Constant[value=1]() # tests/test_passes.py:107:16
  %4 : int = prim::Constant[value=0]() # tests/test_passes.py:107:38
  %5 : int = aten::size(%x.1, %4) # tests/test_passes.py:107:31
  %62 : __torch__.torch.nn.modules.linear.Linear = prim::GetAttr[name="hidden"](%self)
  %63 : int = prim::Constant[value=1]()
  %64 : Float(256:256, 256:1) = prim::GetAttr[name="weight"](%62)
  %65 : Float(256:1) = prim::GetAttr[name="bias"](%62)
  %66 : Float(256:1, 256:256) = aten::t(%64)
  # CHECK: aten::matmul
  %output.1 : Float(1:2560, 10:256, 256:1) = aten::matmul(%x.1, %66)
  %output.3 : Float(1:2560, 10:256, 256:1) = aten::add(%output.1, %65, %63)
  # CHECK: prim::Loop
  %y : Float(1:2560, 10:256, 256:1) = prim::Loop(%5, %8, %x.1) # tests/test_passes.py:107:16
    block0(%i : int, %y.4 : Tensor):
      -> (%8, %output.3)
  return (%y)"""
        FileCheck().run(expect_gstr, str(module._c.forward.graph))


    def test_pass_licm_fail(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                y = [0]
                for i in range(x.size(0)):
                    z = y[0]
                    y.append(z)
                return y
        model = Model()
        module = torch.jit.script(model).eval()
        x = torch.randn(10)
        expected_y = module.forward(x)
        pass_manager._jit_pass_licm(module.graph)
        y = module.forward(x)
        self.assertEqual(y, expected_y)
        # Check getitem op is will not be moved before Loop by licm pass
        expect_gstr = """
graph(%self : __torch__.___torch_mangle_4.Model,
      %x.1 : Tensor):
  %8 : bool = prim::Constant[value=1]() # tests/test_passes.py:101:16
  %2 : int = prim::Constant[value=0]() # tests/test_passes.py:100:21
  %y.1 : int[] = prim::ListConstruct(%2)
  %5 : int = aten::size(%x.1, %2) # tests/test_passes.py:101:31
   # CHECK: prim::Loop
   = prim::Loop(%5, %8) # tests/test_passes.py:101:16
    block0(%i : int):
      # CHECK: aten::__getitem__
      %z.1 : int = aten::__getitem__(%y.1, %2) # tests/test_passes.py:102:24
      %14 : int[] = aten::append(%y.1, %z.1) # tests/test_passes.py:103:20
      -> (%8)
  return (%y.1)"""
        FileCheck().run(expect_gstr, str(module._c.forward.graph))

    def test_hack_cpu_device(self):
        cfg = Config.get_current_context_or_new()
        cfg.enable_force_to_cuda = True

        @torch.jit.script
        def move_to_cpu(x):
            return x.to('cpu')

        with cfg:
            pass_manager._jit_pass_hack_cpu_device(move_to_cpu.graph)
        expect_gstr = """
graph(%x.1 : Tensor):
  %6 : bool = prim::Constant[value=0]()
  %5 : None = prim::Constant()
  # CHECK: prim::Constant[value="cuda"]()
  %18 : Device = prim::Constant[value="cuda"]()
  %8 : Tensor = aten::to(%x.1, %18, %5, %6, %6)
  return (%8)"""
        FileCheck().run(expect_gstr, move_to_cpu.graph)

        @torch.jit.script
        def zeros_cpu():
            return torch.zeros([1, 2], device=torch.device('cpu'))

        with cfg:
            pass_manager._jit_pass_hack_cpu_device(zeros_cpu.graph)
        expect_gstr = """
graph():
  %5 : None = prim::Constant()
  %0 : int = prim::Constant[value=1]()
  %1 : int = prim::Constant[value=2]()
  %2 : int[] = prim::ListConstruct(%0, %1)
  # CHECK: prim::Constant[value="cuda"]()
  %18 : Device = prim::Constant[value="cuda"]()
  %8 : Tensor = aten::zeros(%2, %5, %5, %18, %5)
  return (%8)"""
        FileCheck().run(expect_gstr, zeros_cpu.graph)


if __name__ == "__main__":
    unittest.main()

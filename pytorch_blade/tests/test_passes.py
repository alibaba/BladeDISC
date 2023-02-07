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

    def test_hack_gpu_device_1(self):
        cfg = Config.get_current_context_or_new()
        cfg.force_gpu_constants_to_device = "cuda:2"

        @torch.jit.script
        def move_to_devices(x, y):
            return x.to('cuda'), y.to('cuda:1')

        with cfg:
            pass_manager._jit_pass_hack_gpu_device(move_to_devices.graph)
        expect_gstr = """
graph(%x.1 : Tensor,
      %y.1 : Tensor):
  # CHECK: prim::Constant[value="cuda:2"]
  %57 : Device = prim::Constant[value="cuda:2"]()
  %21 : bool = prim::Constant[value=0]()
  %20 : NoneType = prim::Constant()
  # CHECK: prim::Constant[value="cuda:2"]
  %56 : Device = prim::Constant[value="cuda:2"]()
  %23 : Tensor = aten::to(%x.1, %56, %20, %21, %21)
  %45 : Tensor = aten::to(%y.1, %57, %20, %21, %21)
  %46 : (Tensor, Tensor) = prim::TupleConstruct(%23, %45)
  return (%46)"""
        FileCheck().run(expect_gstr, move_to_devices.graph)

    def test_hack_gpu_device_2(self):
        cfg = Config.get_current_context_or_new()

        @torch.jit.script
        def move_to_devices(x, y):
            return x.to('cuda:1'), y.to('cuda:2')

        with cfg:
            pass_manager._jit_pass_hack_gpu_device(move_to_devices.graph)
        expect_gstr = """
graph(%x.1 : Tensor,
      %y.1 : Tensor):
  # CHECK-DAG: prim::Constant[value="cuda:1"]
  %57 : Device = prim::Constant[value="cuda:1"]()
  %21 : bool = prim::Constant[value=0]()
  %20 : NoneType = prim::Constant()
  # CHECK-DAG: prim::Constant[value="cuda:2"]
  %56 : Device = prim::Constant[value="cuda:2"]()
  %23 : Tensor = aten::to(%x.1, %56, %20, %21, %21)
  %45 : Tensor = aten::to(%y.1, %57, %20, %21, %21)
  %46 : (Tensor, Tensor) = prim::TupleConstruct(%23, %45)
  return (%46)"""
        FileCheck().run(expect_gstr, move_to_devices.graph)


if __name__ == "__main__":
    unittest.main()

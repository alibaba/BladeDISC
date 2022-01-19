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

from torch_blade import utils
from torch_blade.tools import shape_inference
from torch_blade.testing.common_utils import TestCase
from torch.testing import FileCheck

class TestRecordShape(TestCase):

    def test_record_shape(self):

        class TestModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                weight = torch.ones(4, 8)
                self.weight = weight.transpose(0, 1)
                self.bias = torch.ones(4)

            def forward(self, x):
                y = torch.matmul(x, self.weight)
                z = y + self.bias
                return z

        model = torch.jit.script(TestModel())
        example = torch.ones(1, 3, 4, 8)
        shape_inference.record_shape_by_tracing(model._c, (example,))
        expect_graph_str_since_160 = """
        graph(%self : __torch__.TestModel,
              %x.1 : Float(1:96, 3:32, 4:8, 8:1)):
          %7 : int = prim::Constant[value=1]()
          # CHECK: Float(8:1, 4:8) = prim::GetAttr[name="weight"](%self)
          %3 : Float(8:1, 4:8) = prim::GetAttr[name="weight"](%self)
          # CHECK: Float(1:48, 3:16, 4:4, 4:1) = aten::matmul
          %y.1 : Float(1:48, 3:16, 4:4, 4:1) = aten::matmul(%x.1, %3) # tests/test_record_shape.py:22:20
          %6 : Float(4:1) = prim::GetAttr[name="bias"](%self)
          # CHECK: Float(1:48, 3:16, 4:4, 4:1) = aten::add
          %z.1 : Float(1:48, 3:16, 4:4, 4:1) = aten::add(%y.1, %6, %7) # tests/test_record_shape.py:23:20
          return (%z.1)"""
        expect_graph_str_since_171 = """
        graph(%self : __torch__.TestModel,
              %x.1 : Float(1:96, 3:32, 4:8, 8:1, requires_grad=0, device=cpu)):
          %7 : int = prim::Constant[value=1]()
          # CHECK: Float(8:1, 4:8, requires_grad=0, device=cpu)
          %3 : Float(8:1, 4:8, requires_grad=0, device=cpu) = prim::GetAttr[name="weight"](%self)
          # CHECK: Float(1:48, 3:16, 4:4, 4:1, requires_grad=0, device=cpu) = aten::matmul
          %y.1 : Float(1:48, 3:16, 4:4, 4:1, requires_grad=0, device=cpu) = aten::matmul(%x.1, %3) # tests/test_record_shape.py:22:20
          %6 : Float(4:1, requires_grad=0, device=cpu) = prim::GetAttr[name="bias"](%self)
          # CHECK: Float(1:48, 3:16, 4:4, 4:1, requires_grad=0, device=cpu) = aten::add
          %z.1 : Float(1:48, 3:16, 4:4, 4:1, requires_grad=0, device=cpu) = aten::add(%y.1, %6, %7) # tests/test_record_shape.py:23:20
          return (%z.1)"""
        expect_graph_str_since_181 = """
        graph(%self : __torch__.TestModel,
              %x.1 : Float(1, 3, 4, 8, strides=[96, 32, 8, 1], requires_grad=0, device=cpu)):
          %7 : int = prim::Constant[value=1]()
          # CHECK: Float(8, 4, strides=[1, 8], requires_grad=0, device=cpu) = prim::GetAttr[name="weight"](%self)
          %3 : Float(8, 4, strides=[1, 8], requires_grad=0, device=cpu) = prim::GetAttr[name="weight"](%self)
          # CHECK: Float(1, 3, 4, 4, strides=[48, 16, 4, 1], requires_grad=0, device=cpu) = aten::matmul
          %y.1 : Float(1, 3, 4, 4, strides=[48, 16, 4, 1], requires_grad=0, device=cpu) = aten::matmul(%x.1, %3) # tests/test_record_shape.py:23:20
          %6 : Float(4, strides=[1], requires_grad=0, device=cpu) = prim::GetAttr[name="bias"](%self)
          # CHECK: Float(1, 3, 4, 4, strides=[48, 16, 4, 1], requires_grad=0, device=cpu) = aten::add
          %z.1 : Float(1, 3, 4, 4, strides=[48, 16, 4, 1], requires_grad=0, device=cpu) = aten::add(%y.1, %6, %7) # tests/test_record_shape.py:24:20
          return (%z.1)"""
        expect_graph_str = expect_graph_str_since_171 if utils.torch_version_number() >= "1.7.1" else expect_graph_str_since_160
        expect_graph_str = expect_graph_str_since_181 if utils.torch_version_number() >= "1.8.1" else expect_graph_str
        FileCheck().run(expect_graph_str, model._c.forward.graph)

    def test_unintialized(self):
        class TestModel(torch.nn.Module):

            def forward(self, x):
                if x.numel() > 0:
                    y = x
                else:
                    raise Exception("error")
                return y

        model = torch.jit.script(TestModel())
        expect_str = "# CHECK: Tensor = prim::Uninitialized()"
        FileCheck().run(expect_str, model.forward.graph)

        example = torch.ones(1, 3, 4, 8)
        shape_inference.record_shape_by_tracing(model._c, (example,))
        expect_gstr_since_160 = '''
        graph(%self : __torch__.___torch_mangle_0.TestModel,
              %x.1 : Float(1:96, 3:32, 4:8, 8:1)):
          %7 : str = prim::Constant[value="Exception"]() # tests/test_record_shape.py:51:20
          %4 : int = prim::Constant[value=0]() # tests/test_record_shape.py:48:31
          # CHECK: Tensor = prim::Uninitialized()
          %20 : Tensor = prim::Uninitialized()
          %3 : int = aten::numel(%x.1) # tests/test_record_shape.py:48:19
          %5 : bool = aten::gt(%3, %4) # tests/test_record_shape.py:48:19
          # CHECK: Float(1:96, 3:32, 4:8, 8:1) = prim::If(%5)
          %y : Float(1:96, 3:32, 4:8, 8:1) = prim::If(%5) # tests/test_record_shape.py:48:16
            block0():
              -> (%x.1)
            block1():
               = prim::RaiseException(%7) # tests/test_record_shape.py:51:20
              -> (%20)
          return (%y)'''
        expect_gstr_since_171 = '''
        graph(%self : __torch__.___torch_mangle_0.TestModel,
              %x.1 : Float(1:96, 3:32, 4:8, 8:1, requires_grad=0, device=cpu)):
          %7 : str = prim::Constant[value="error"]() # tests/test_record_shape.py:64:36
          %4 : int = prim::Constant[value=0]() # tests/test_record_shape.py:61:31
          # CHECK: Tensor = prim::Uninitialized()
          %20 : Tensor = prim::Uninitialized()
          %3 : int = aten::numel(%x.1) # tests/test_record_shape.py:61:19
          %5 : bool = aten::gt(%3, %4) # tests/test_record_shape.py:61:19
          # CHECK: Float(1:96, 3:32, 4:8, 8:1, requires_grad=0, device=cpu) = prim::If(%5)
          %y : Float(1:96, 3:32, 4:8, 8:1, requires_grad=0, device=cpu) = prim::If(%5)
            block0():
              -> (%x.1)
            block1():
               = prim::RaiseException(%7) # tests/test_record_shape.py:64:20
              -> (%20)
          return (%y)'''

        expect_gstr_since_181 = '''
        graph(%self : __torch__.___torch_mangle_0.TestModel,
              %x.1 : Float(1, 3, 4, 8, strides=[96, 32, 8, 1], requires_grad=0, device=cpu)):
          %8 : str = prim::Constant[value="error"]() # tests/test_record_shape.py:77:36
          %4 : int = prim::Constant[value=0]() # tests/test_record_shape.py:74:31
          # CHECK: Tensor = prim::Uninitialized()
          %21 : Tensor = prim::Uninitialized()
          %3 : int = aten::numel(%x.1) # tests/test_record_shape.py:74:19
          %5 : bool = aten::gt(%3, %4) # tests/test_record_shape.py:74:19
          # CHECK: Float(1, 3, 4, 8, strides=[96, 32, 8, 1], requires_grad=0, device=cpu) = prim::If(%5)
          %y : Float(1, 3, 4, 8, strides=[96, 32, 8, 1], requires_grad=0, device=cpu) = prim::If(%5)
            block0():
              -> (%x.1)
            block1():
               = prim::RaiseException(%8) # tests/test_record_shape.py:77:20
              -> (%21)
          return (%y)'''
        expect_gstr = expect_gstr_since_171 if utils.torch_version_number() >= "1.7.1" else expect_gstr_since_160
        expect_gstr = expect_gstr_since_181 if utils.torch_version_number() >= "1.8.1" else expect_gstr
        FileCheck().run(expect_gstr, model._c.forward.graph)


if __name__ == '__main__':
    unittest.main()

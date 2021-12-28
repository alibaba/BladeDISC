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
from torch_blade import mlir

from tests.mlir.testing_utils import DiscTestCase


class TestDiscList(DiscTestCase):

    def test_list_int(self):
        @torch.jit.script
        def return_const_scalar():
            return torch.tensor([1, 2, 3, 4])

        return_const = self._ScriptFunction2Module(return_const_scalar)
        graph = return_const.forward.graph
        graph.eraseInput(0)
        disc_bytes, _, _, _ = mlir.cvt_torchscript_to_mhlo(graph)
        expect_str = """
          module  {
            func @main() -> tensor<4xi64> attributes {tf.entry_function = {input_placements = "", inputs = "", output_placements = "cpu", outputs = "7"}} {
              %false = constant false loc(#loc0)
              %c1_i64 = constant 1 : i64 loc(#loc1)
              %c2_i64 = constant 2 : i64 loc(#loc2)
              %c3_i64 = constant 3 : i64 loc(#loc3)
              %c4_i64 = constant 4 : i64 loc(#loc4)
              # CHECK: tensor.from_elements
              %0 = tensor.from_elements %c1_i64, %c2_i64, %c3_i64, %c4_i64 : tensor<4xi64> loc(#loc5)
              return %0 : tensor<4xi64> loc(#loc0)
            } loc(#loc0)
          } loc(#loc0)
        """
        FileCheck().run(expect_str, disc_bytes)

    def test_list_bool(self):
        gstr = """graph():
                    %0 : None = prim::Constant()
                    %1 : int = prim::Constant[value=0]()
                    %2 : int = prim::Constant[value=1]()
                    %3 : bool = prim::Constant[value=0]()
                    %4 : bool = prim::Constant[value=1]()
                    %list : bool[] = prim::ListConstruct(%3, %4)
                    %w : bool = aten::__getitem__(%list, %1)
                    %z : bool = aten::__getitem__(%list, %2)
                    %r : Tensor = aten::tensor(%1, %0, %0, %3)
                    return (%r)"""
        graph = torch._C.parse_ir(gstr)
        disc_bytes, _, _, _ = mlir.cvt_torchscript_to_mhlo(graph)
        expect_str = """
          module  {
            func @main() -> tensor<i64> attributes {tf.entry_function = {input_placements = "", inputs = "", output_placements = "cpu", outputs = "8"}} {
              # CHECK: constant 0
              %c0_i64 = constant 0 : i64 loc(#loc)
              # CHECK: constant 1
              %c1_i64 = constant 1 : i64 loc(#loc)
              # CHECK: constant false
              %false = constant false loc(#loc)
              # CHECK: constant true 
              %true = constant true loc(#loc)
              %0 = mhlo.constant dense<0> : tensor<i64> loc(#loc)
              return %0 : tensor<i64> loc(#loc)
            } loc(#loc)
          } loc(#loc)
        """
        FileCheck().run(expect_str, disc_bytes)

    def test_list_float(self):
        gstr = """graph():
                    %0 : None = prim::Constant()
                    %1 : int = prim::Constant[value=0]()
                    %2 : int = prim::Constant[value=1]()
                    %3 : float = prim::Constant[value=3.]()
                    %4 : float = prim::Constant[value=4.]()
                    %5 : bool = prim::Constant[value=0]()
                    %list : float[] = prim::ListConstruct(%3, %4)
                    %w : float = aten::__getitem__(%list, %1)
                    %z : float = aten::__getitem__(%list, %2)
                    %r : Tensor = aten::tensor(%1, %0, %0, %5)
                    return (%r)"""
        graph = torch._C.parse_ir(gstr)
        disc_bytes, _, _, _ = mlir.cvt_torchscript_to_mhlo(graph)
        expect_str = """
          module  {
            func @main() -> tensor<i64> attributes {tf.entry_function = {input_placements = "", inputs = "", output_placements = "cpu", outputs = "9"}} {
              %c0_i64 = constant 0 : i64 loc(#loc)
              %c1_i64 = constant 1 : i64 loc(#loc)
              # CHECK: constant 3.000000e+00 : f64
              %cst = constant 3.000000e+00 : f64 loc(#loc)
              # CHECK: constant 4.000000e+00 : f64
              %cst_0 = constant 4.000000e+00 : f64 loc(#loc)
              %false = constant false loc(#loc)
              %0 = mhlo.constant dense<0> : tensor<i64> loc(#loc)
              return %0 : tensor<i64> loc(#loc)
            } loc(#loc)
          } loc(#loc)
          #loc = loc(unknown)
        """
        FileCheck().run(expect_str, disc_bytes)

    def test_list_tensor(self):
        gstr = """graph(%x.1 : Float(4, 4),
                        %y.1 : Float(4, 4)):
                    %2 : int = prim::Constant[value=1]()
                    %3 : int = prim::Constant[value=0]()
                    %list_tensors.1 : Tensor[] = prim::ListConstruct(%x.1, %y.1)
                    %w.1 : Tensor = aten::__getitem__(%list_tensors.1, %2)
                    %z.1 : Tensor = aten::__getitem__(%list_tensors.1, %3)
                    return (%w.1, %z.1)"""
        graph = torch._C.parse_ir(gstr)
        disc_bytes, _, _, _ = mlir.cvt_torchscript_to_mhlo(graph)
        expect_str = """
          #loc = loc(unknown)
          module  {
            func @main(%arg0: tensor<?x?xf32> loc(unknown), %arg1: tensor<?x?xf32> loc(unknown)) -> (tensor<?x?xf32>, tensor<?x?xf32>) attributes {tf.entry_function = {input_placements = "cpu,cpu", inputs = "x.1,y.1", output_placements = "cpu,cpu", outputs = "5,6"}} {
              %c1_i64 = constant 1 : i64 loc(#loc)
              %c0_i64 = constant 0 : i64 loc(#loc)
              # CHECK: return %arg1, %arg0
              return %arg1, %arg0 : tensor<?x?xf32>, tensor<?x?xf32> loc(#loc)
            } loc(#loc)
          } loc(#loc)
        """
        FileCheck().run(expect_str, disc_bytes)

if __name__ == "__main__":
    unittest.main()

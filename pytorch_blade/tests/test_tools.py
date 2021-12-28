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
from torch_blade import tools
from torch_blade.testing.common_utils import Feedforward, TestCase


class TestTools(TestCase):

    def setUp(self):
        ff_net = Feedforward(64, 10)
        self.ff_net = torch.jit.script(ff_net)

    def test_add_method(self):
        self.ff_net._c.create_method_from_graph('inference', self.ff_net.graph)
        # pylint: disable=no-member
        dummy_input = torch.ones([64, 64])
        fwd_output = self.ff_net.forward(dummy_input)
        inf_output = self.ff_net.inference(dummy_input)
        self.assertEqual(fwd_output, inf_output)

    def test_add_function_has_input(self):
        def relu(x: torch.Tensor):
            # pylint: disable=no-member
            return torch.relu(x)

        relu = torch.jit.script(relu)
        self.ff_net._c.create_method_from_graph('relu', relu.graph)
        # pylint: disable=no-member
        dummy_input = torch.randn([64, 64])
        inf_output = self.ff_net.relu(dummy_input)
        fwd_output = self.ff_net.relu.forward(dummy_input)
        self.assertEqual(fwd_output, inf_output)

    def test_add_function_non_input(self):
        def create_input():
            # pylint: disable=no-member
            return torch.ones([64, 64])
        create_input = torch.jit.script(create_input)
        self.ff_net._c.create_method_from_graph(
            'create_input', create_input.graph)
        # pylint: disable=no-member
        dummy_input = self.ff_net.create_input()
        self.assertEqual(dummy_input, torch.ones([64, 64]))

    def test_node_schema_str(self):
        @torch.jit.script
        def shape(x):
            return x.size()[1]

        expect_gstr = '''
        graph(%x.1 : Tensor):
          # CHECK: prim::Constant
          %3 : int = prim::Constant[value=1]() # tests/test_tools.py:49:28
          # CHECK: aten::size
          %2 : int[] = aten::size(%x.1) # tests/test_tools.py:49:19
          # CHECK: aten::__getitem__
          %4 : int = aten::__getitem__(%2, %3) # tests/test_tools.py:49:19
          return (%4)'''

        FileCheck().run(expect_gstr, shape.graph)
        schemas = [tools.node_schema_str(n) for n in shape.graph.nodes()]
        self.assertEqual(schemas[0], "")
        self.assertEqual(schemas[1], "aten::size(Tensor self) -> (int[])")
        self.assertEqual(schemas[2], "aten::__getitem__.t(t[](a) list, int idx) -> (t(*))")

    def test_from_number_type(self):

        def _test_type(scalar_type_str):
            gstr = f"""
            graph(%x.1 : {scalar_type_str}):
              return (%x.1)"""

            type_map = {"int": "Long", "bool": "Bool", "float": "Float"}
            graph = torch._C.parse_ir(gstr)
            value = graph.input_list()[0]
            tools.cast_to_tensor_type(value)
            self.assertTrue(value.isCompleteTensor())
            self.assertTrue(value.type().scalarType() == type_map[scalar_type_str])

        _test_type("int")
        _test_type("bool")
        _test_type("float")

    def test_freeze_module(self):
        class Sparse(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self._output = torch.randn([1, 3, 224, 224])

            def forward(self, x):
                return self._output

        class Model(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.sparse = Sparse()
                self.model = torch.nn.ReLU()

            def forward(self, x):
                y = self.sparse(x)
                return self.model(y)

        m = torch.jit.script(Model())
        m.eval()

        m0 = tools.freeze_module(m._c, disableShapePeephole=True, preservedAttrs=[])
        m1 = tools.freeze_module(m._c, disableShapePeephole=True, preservedAttrs=["sparse"])

        m0_expect_gstr = """
        graph(%self : __torch__.___torch_mangle_1.Model,
              %x.1 : Tensor):
          # CHECK-NOT: Sparse = prim::GetAttr[name="sparse"](%self)
          # CHECK-NOT: Tensor = aten::relu
          %11 : Float(1:150528, 3:50176, 224:224, 224:1) = prim::Constant[value=<Tensor>]()
          return (%11)"""

        m1_expect_gstr = """
        graph(%self : __torch__.___torch_mangle_4.Model,
              %x.1 : Tensor):
          # CHECK: Sparse = prim::GetAttr[name="sparse"](%self)
          %2 : __torch__.___torch_mangle_5.Sparse = prim::GetAttr[name="sparse"](%self)
          # CHECK: %y.1 : Tensor = prim::GetAttr[name="_output"](%2)
          %y.1 : Tensor = prim::GetAttr[name="_output"](%2)
          # CHECK: Tensor = aten::relu
          %result.2 : Tensor = aten::relu(%y.1)
          return (%result.2)"""

        FileCheck().run(m0_expect_gstr, m0.forward.graph)
        FileCheck().run(m1_expect_gstr, m1.forward.graph)


if __name__ == '__main__':
    unittest.main()

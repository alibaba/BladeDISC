import unittest
import torch
from torch.testing import FileCheck
from torch_addons.python_ir_analysis import _jit_pass_clean_python_ir
from torch_addons.testing.common_utils import TestCase


class TestPythonIrPass(TestCase):

    def test_list_append0(self):
        @torch.jit.script
        def list_append(x):
            tensors = [x]
            y = tensors[0] + 1
            tensors.append(y)
            z = tensors[1] + 1
            tensors.append(z)
            return tensors[-1]

        _jit_pass_clean_python_ir(list_append.graph)
        expect_gstr = """
        graph(%x.1 : Tensor):
          %6 : int = prim::Constant[value=1]()
          # CHECK-NOT: aten::append 
          # CHECK-NOT: aten::__getitem__
          # CHECK-COUNT-EXACTLY-2: aten::add
          %y.1 : Tensor = aten::add(%x.1, %6, %6)
          %z.1 : Tensor = aten::add(%y.1, %6, %6)
          return (%z.1)"""
        FileCheck().run(expect_gstr, list_append.graph)

    def test_list_append1(self):
        @torch.jit.script
        def list_append(x):
            tensors = [x]
            y = tensors[0] + 1
            tensors.append(y)
            z = tensors[1] + 1
            tensors.append(z)
            return tensors

        _jit_pass_clean_python_ir(list_append.graph)
        expect_gstr = """
        graph(%x.1 : Tensor):
          # CHECK-NOT: aten::append 
          # CHECK-NOT: aten::__getitem__
          # CHECK-COUNT-EXACTLY-2: aten::add
          %6 : int = prim::Constant[value=1]()
          %y.1 : Tensor = aten::add(%x.1, %6, %6)
          %z.1 : Tensor = aten::add(%y.1, %6, %6)
          # CHECK-COUNT-EXACTLY-1: prim::ListConstruct
          %30 : Tensor[] = prim::ListConstruct(%x.1, %y.1, %z.1)
          return (%30)"""
        FileCheck().run(expect_gstr, list_append.graph)

    def test_list_append2(self):
        @torch.jit.script
        def list_append(x, iter: int):
            tensors = [x]
            y = tensors[0] + 1
            tensors.append(y)
            for k in range(iter):
                y = tensors[k] + k
                tensors.append(y)
            return tensors

        _jit_pass_clean_python_ir(list_append.graph)
        expect_gstr = """
        graph(%x.1 : Tensor,
              %iter.1 : int):
          %16 : bool = prim::Constant[value=1]()
          %7 : int = prim::Constant[value=1]()
          # CHECK-COUNT-EXACTLY-1: aten::add
          %y.1 : Tensor = aten::add(%x.1, %7, %7)
          # CHECK-COUNT-EXACTLY-1: prim::ListConstruct
          %47 : Tensor[] = prim::ListConstruct(%x.1, %y.1)
          %42 : int = prim::Constant[value=0]()
          # CHECK-COUNT-EXACTLY-1: prim::Loop
          %43 : int = prim::Loop(%iter.1, %16, %42)
            block0(%k.1 : int, %44 : int):
          # CHECK-COUNT-EXACTLY-1: aten::__getitem__
          # CHECK-COUNT-EXACTLY-1: aten::append
              %20 : Tensor = aten::__getitem__(%47, %44)
              %y.3 : Tensor = aten::add(%20, %44, %7)
              %27 : Tensor[] = aten::append(%47, %y.3)
              %45 : int = prim::Constant[value=1]()
              %46 : int = aten::add(%44, %45)
              -> (%16, %46)
          return (%47)"""
        FileCheck().run(expect_gstr, list_append.graph)

    def test_list_extend(self):
        @torch.jit.script
        def list_extend0(x):
            scales: List[float] = []
            scales.append(2.0)
            scales.append(2.0)
            scales.extend(scales)
            return scales

        _jit_pass_clean_python_ir(list_extend0.graph)
        expect_gstr = """
        graph(%x : Tensor):
        # CHECK: float[] = prim::Constant[value=[2., 2., 2., 2.]]()
          %23 : float[] = prim::Constant[value=[2., 2., 2., 2.]]()
          return (%23)"""
        FileCheck().run(expect_gstr, list_extend0.graph)

        @torch.jit.script
        def list_extend1(x):
            scales: List[float] = []
            scales.append(2.0)
            scales.append(2.0)
            if len(scales) > 3:
                d = scales[:]
            else:
                d = [4.0, 5.0]
            scales.extend(d)
            return scales

        _jit_pass_clean_python_ir(list_extend1.graph)
        expect_gstr = """
        graph(%x : Tensor):
        # CHECK: float[] = prim::Constant[value=[2., 2., 4., 5.]]()
          %23 : float[] = prim::Constant[value=[2., 2., 4., 5.]]()
          return (%23)"""
        FileCheck().run(expect_gstr, list_extend1.graph)

    def test_list_insert0(self):
        @torch.jit.script
        def list_insert(x):
            tensors = [x]
            y = tensors[0] + 1
            tensors.insert(0, y)
            z = tensors[1] + 1
            tensors.insert(1, z)
            # exactly return x
            return tensors[-1]

        _jit_pass_clean_python_ir(list_insert.graph)
        expect_gstr = """
        graph(%x.1 : Tensor):
          # CHECK-NOT: aten::append 
          # CHECK-NOT: aten::__getitem__
          # CHECK-NOT: aten::add
          return (%x.1)"""
        FileCheck().run(expect_gstr, list_insert.graph)

    def test_list_insert1(self):
        @torch.jit.script
        def list_insert(x):
            tensors = [x]
            y = tensors[0] + 1
            tensors.insert(0, y)
            z = tensors[0] + 1
            tensors.insert(1, z)
            return tensors[1]

        _jit_pass_clean_python_ir(list_insert.graph)
        expect_gstr = """
        graph(%x.1 : Tensor):
          %6 : int = prim::Constant[value=1]()
          # CHECK-NOT: aten::append 
          # CHECK-NOT: aten::__getitem__
          # CHECK-COUNT-EXACTLY-2: aten::add
          %y.1 : Tensor = aten::add(%x.1, %6, %6)
          %z.1 : Tensor = aten::add(%y.1, %6, %6)
          return (%z.1)"""
        FileCheck().run(expect_gstr, list_insert.graph)

    def test_list_insert2(self):
        @torch.jit.script
        def list_insert(x):
            tensors = [x]
            y = tensors[0] + 1
            tensors.insert(0, y)
            z = tensors[1] + 1
            tensors.insert(1, z)
            return tensors
        _jit_pass_clean_python_ir(list_insert.graph)
        expect_gstr = """
        graph(%x.1 : Tensor):
          %6 : int = prim::Constant[value=1]()
          # CHECK-COUNT-EXACTLY-2: aten::add
          %y.1 : Tensor = aten::add(%x.1, %6, %6)
          %z.1 : Tensor = aten::add(%x.1, %6, %6)
          # CHECK-COUNT-EXACTLY-1: prim::ListConstruct
          %30 : Tensor[] = prim::ListConstruct(%y.1, %z.1, %x.1)
          return (%30)"""
        FileCheck().run(expect_gstr, list_insert.graph)

    def test_dict_setitem(self):
        @torch.jit.script
        def dict_setitem(x):
            tensors = {'x': x}
            y = tensors['x'] + 1
            tensors['y'] = y
            z = tensors['y'] + 1
            tensors['z'] = z
            return tensors['z']

        _jit_pass_clean_python_ir(dict_setitem.graph)
        expect_gstr = """
        graph(%x.1 : Tensor):
          %6 : int = prim::Constant[value=1]()
          # CHECK-NOT: aten::_set_item
          # CHECK-NOT: aten::__getitem__
          # CHECK-COUNT-EXACTLY-2: aten::add
          %y.1 : Tensor = aten::add(%x.1, %6, %6)
          %z.1 : Tensor = aten::add(%y.1, %6, %6)
          return (%z.1)"""
        FileCheck().run(expect_gstr, dict_setitem.graph)

    def test_dict_keys(self):
        @torch.jit.script
        def dict_keys(x):
            tensors = {'x': x}
            y = tensors['x'] + 1
            tensors['y'] = y
            z = tensors['y'] + 1
            tensors['z'] = z
            return tensors.keys()

        _jit_pass_clean_python_ir(dict_keys.graph)
        expect_gstr = """
        graph(%x.1 : Tensor):
        # CHECK: str[] = prim::Constant[value=["x", "y", "z"]]()
          %38 : str[] = prim::Constant[value=["x", "y", "z"]]()
          return (%38)"""
        FileCheck().run(expect_gstr, dict_keys.graph)


    def test_list_contains(self):
        @torch.jit.script
        def list_contains(x):
            tensors = ['y']
            tensors.append('x')
            if 'x' in tensors:
                y = x + 1
            else:
                y = x
            return y
        _jit_pass_clean_python_ir(list_contains.graph)

        expect_gstr = """
        graph(%x.1 : Tensor):
          # CHECK-NOT: aten::__contains__
          # CHECK-COUNT-EXACTLY-1: aten::add
          %10 : int = prim::Constant[value=1]()
          %y.1 : Tensor = aten::add(%x.1, %10, %10)
          return (%y.1)"""
        FileCheck().run(expect_gstr, list_contains.graph)

    def test_dict_contains(self):
        @torch.jit.script
        def dict_contains(x):
            tensors = {'x': x}
            if 'x' in tensors:
                y = tensors['x'] + 1
            else:
                y = x
            if 'y' not in tensors:
                tensors['y'] = y
            z = tensors['y'] + 1
            if len(tensors) > 1:
                tensors['z'] = z
            return tensors['z']
        _jit_pass_clean_python_ir(dict_contains.graph)
        expect_gstr = """
        graph(%x.1 : Tensor):
          %6 : int = prim::Constant[value=1]()
          # CHECK-NOT: prim::If
          # CHECK-NOT: aten::_set_item
          # CHECK-NOT: aten::__getitem__
          # CHECK-NOT: aten::__contains__
          # CHECK-NOT: aten::len
          # CHECK-COUNT-EXACTLY-2: aten::add
          %y.1 : Tensor = aten::add(%x.1, %6, %6)
          %z.1 : Tensor = aten::add(%y.1, %6, %6)
          return (%z.1)"""
        FileCheck().run(expect_gstr, dict_contains.graph)

    def test_loop_unroll(self):
        @torch.jit.script
        def loop_unroll(x):
            tensors = [x]
            tensor_dict = {'x': x}
            for k in range(4):
                y = tensors[k] + tensor_dict['x']
                tensors.append(y)
                tensor_dict['x'] = y
            return tensors[1], tensor_dict['x']

        _jit_pass_clean_python_ir(loop_unroll.graph)

        expect_gstr = """
        graph(%x.1 : Tensor):
          # CHECK-NOT: aten::_set_item
          # CHECK-NOT: aten::__getitem__
          # CHECK-COUNT-EXACTLY-4: aten::add
          %27 : int = prim::Constant[value=1]()
          %y.1 : Tensor = aten::add(%x.1, %x.1, %27)
          %y.2 : Tensor = aten::add(%y.1, %y.1, %27)
          %y.3 : Tensor = aten::add(%y.2, %y.2, %27)
          %y.4 : Tensor = aten::add(%y.3, %y.3, %27)
          %32 : (Tensor, Tensor) = prim::TupleConstruct(%y.1, %y.4)
          return (%32)"""

if __name__ == "__main__":
    unittest.main()

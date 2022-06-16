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

from tests.disc.testing_base import DiscTestCase, skipIfEnableTorchMlir, isTorchMlirEnable


class TestDiscBinaryOps(DiscTestCase):
    def _check_type(self, out, res):
        if isinstance(out, torch.Tensor):
            self.assertTrue(isinstance(res, torch.Tensor))
            self.assertEqual(out.dtype, res.dtype)
        elif isinstance(out, tuple):
            self.assertTrue(isinstance(res, tuple))
            for o, r in zip(out, res):
                self._check_type(o, r)
        else:
            # currently, only support return type of Tensor and Tuple[Tensor]
            self.assertTrue(False)

    def _test_binary_type_promotion(self, torch_func):
        @torch.jit.script
        def func(x, y):
            return torch_func(x, y)

        # test type cast int32 -> float32
        x = torch.randn([4, 4], dtype=torch.float32, device=self.device)
        y = torch.randint(3, [4, 4], dtype=torch.int32, device=self.device)
        test_data = (x, y)
        out, res = self._test_cvt_to_disc(func, test_data)
        self._check_type(out, res)

        # test type promote float32 -> float64
        x = torch.randn([4, 4], dtype=torch.float32, device=self.device)
        y = torch.randn([4, 4], dtype=torch.float64, device=self.device)
        test_data = (x, y)
        out, res = self._test_cvt_to_disc(func, test_data)
        self._check_type(out, res)

    def _test_binary_ops(self, binary_ops_func, test_int=True):
        # test no broadcast
        x = torch.randn([10, 2, 3, 4], device=self.device)
        y = torch.randn([10, 2, 3, 4], device=self.device)
        test_data = (x, y)
        out, res = self._test_cvt_to_disc(binary_ops_func, test_data)
        self._check_type(out, res)

        # test broadcast in-dims
        x = torch.randn([10, 2, 3, 4], device=self.device)
        y = torch.randn([1, 2, 1, 4], device=self.device)
        test_data = (x, y)
        out, res = self._test_cvt_to_disc(binary_ops_func, test_data)
        self._check_type(out, res)

        # test lower rank to higher rank broadcast
        x = torch.randn([10, 2, 3, 4], device=self.device)
        y = torch.randn([4], device=self.device)
        test_data = (x, y)
        out, res = self._test_cvt_to_disc(binary_ops_func, test_data)
        self._check_type(out, res)

        # test scalar Schema
        x = torch.randn([10, 2, 3, 4], device=self.device)
        y = torch.tensor(2.0, device=self.device)
        test_data = (x, y)
        out, res = self._test_cvt_to_disc(binary_ops_func, test_data)
        self._check_type(out, res)

        # test integer
        if test_int:
            x = torch.randint(1, 3, [10, 2], device=self.device)
            y = torch.randint(1, 3, [10, 2], device=self.device)
            test_data = (x, y)
            out, res = self._test_cvt_to_disc(binary_ops_func, test_data)
            self._check_type(out, res)

    def _test_func(self, torch_func, test_int=True):
        @torch.jit.script
        def func1(x, y):
            return torch_func(x, y.expand_as(x))

        self._test_binary_ops(func1, test_int)

        @torch.jit.script
        def func2(x, y):
            return torch_func(x, y)

        self._test_binary_ops(func2, test_int)

    def _test_scalar_func(self, torch_func):
        @torch.jit.script
        def func(x, y):
            a = x.size(0)
            b = y.size(0)
            c = torch_func(a, b)
            z = torch_func(x, c)
            return torch_func(z, y)

        # test scalar Schema
        x = torch.randn([10, 2, 3, 4], device=self.device)
        test_data = (x, x)
        out, res = self._test_cvt_to_disc(func, test_data)
        self._check_type(out, res)

    def _test_rhs_scalar_func(self, torch_func):
        @torch.jit.script
        def func(x, y):
            a = x.size(0)
            z = torch_func(x, a)
            return torch_func(z, y)

        # test scalar Schema
        x = torch.randn([10, 2, 3, 4], device=self.device)
        test_data = (x, x)
        out, res = self._test_cvt_to_disc(func, test_data)
        self._check_type(out, res)

    def _test_func_has_alpha(self, torch_func):
        @torch.jit.script
        def func1(x, y):
            return torch_func(x, y.expand_as(x), alpha=0.1)

        self._test_binary_ops(func1, False)

        @torch.jit.script
        def func2(x, y):
            return torch_func(x, y, alpha=0.1)

        self._test_binary_ops(func2, False)

        @torch.jit.script
        def func3(x, y):
            return torch_func(x, y, alpha=1)

        self._test_binary_ops(func3)

    def test_binary_type_promotion(self):
        self._test_binary_type_promotion(torch.sub)
        self._test_binary_type_promotion(torch.add)
        self._test_binary_type_promotion(torch.mul)
        self._test_binary_type_promotion(torch.true_divide)
        self._test_binary_type_promotion(torch.floor_divide)
        if not isTorchMlirEnable():
            self._test_binary_type_promotion(torch.rsub)
            self._test_func(torch.logical_and)
            self._test_func(torch.logical_or)

    def test_arithmetic_func(self):
        self._test_func(torch.sub)
        self._test_func(torch.add)
        self._test_func(torch.mul)
        # torch.aten.div between Tensor[int]s meet RefineType error in TorchMLIR
        self._test_func(torch.true_divide, test_int=not isTorchMlirEnable())
        self._test_func(torch.floor_divide, test_int=not isTorchMlirEnable())
        if not isTorchMlirEnable():
            self._test_func(torch.rsub)
            self._test_func(torch.logical_and)
            self._test_func(torch.logical_or)
        # skip the test because disc compilation failed, refs to debug logs:
        # https://bladedisc-ci.oss-cn-hongkong.aliyuncs.com/download/debug/dump_dir.logic_xor.0516.tar.gz
        # self._test_func(torch.logical_xor)

    def test_arithmetic_func_has_alpha(self):
        self._test_func_has_alpha(torch.sub)
        self._test_func_has_alpha(torch.add)
        if not isTorchMlirEnable():
          self._test_func_has_alpha(torch.rsub)

    def test_scalar_arithmetic_func(self):
        self._test_scalar_func(torch.sub)
        self._test_scalar_func(torch.add)
        self._test_scalar_func(torch.mul)
        if not isTorchMlirEnable():
            self._test_scalar_func(torch.div)
            self._test_rhs_scalar_func(torch.rsub)
            self._test_func(torch.logical_and)
            self._test_func(torch.logical_or)

    def _test_logic_func(self, torch_func):
        @torch.jit.script
        def func(x, y):
            return torch_func(x, y)

        x = torch.randint(0, 2, [4, 4], device=self.device).bool()
        y = torch.randint(0, 2, [4, 4], device=self.device).bool()
        test_data = (x, y)
        out, res = self._test_cvt_to_disc(func, test_data)
        self._check_type(out, res)

        @torch.jit.script
        def func(x, y):
            return torch_func(x, y)

        y = torch.randint(0, 2, [4], device=self.device).bool()
        test_data = (x, y)
        out, res = self._test_cvt_to_disc(func, test_data)
        self._check_type(out, res)

    def test_logic_funcs(self):

        @torch.jit.script
        def logical_and(x, y):
            return x & y

        self._test_logic_func(logical_and)

        @torch.jit.script
        def logical_or(x, y):
            return x | y

        if not isTorchMlirEnable():
            self._test_logic_func(logical_or)

    @skipIfEnableTorchMlir()
    def test_pow(self):
        self._test_func(torch.pow)
        self._test_scalar_func(torch.pow)

    def _test_cmp_func(self, torch_func):
        @torch.jit.script
        def func(x, y):
            return torch_func(x, 2.0), torch_func(y, 2.0)
        self._test_func(func)

    @skipIfEnableTorchMlir()
    def test_gt(self):
        self._test_func(torch.gt)
        self._test_binary_type_promotion(torch.gt)
        self._test_cmp_func(torch.gt)

    @skipIfEnableTorchMlir()
    def test_ge(self):
        self._test_func(torch.ge)
        self._test_binary_type_promotion(torch.ge)
        self._test_cmp_func(torch.ge)

    def test_eq(self):
        self._test_func(torch.eq)
        self._test_binary_type_promotion(torch.eq)
        self._test_cmp_func(torch.eq)

    def test_ne(self):
        self._test_func(torch.ne)
        self._test_binary_type_promotion(torch.ne)
        self._test_cmp_func(torch.ne)

    @skipIfEnableTorchMlir()
    def test_le(self):
        self._test_func(torch.le)
        self._test_binary_type_promotion(torch.le)
        self._test_cmp_func(torch.le)

    def test_lt(self):
        self._test_func(torch.lt)
        self._test_binary_type_promotion(torch.lt)
        self._test_cmp_func(torch.lt)

    @skipIfEnableTorchMlir()
    def test_arange(self):
        # Given int dtype.
        @torch.jit.script
        def func_int(x):
            return torch.arange(x, dtype=torch.int)
        test_data = (torch.tensor(10),)
        out, res = self._test_cvt_to_disc(func_int, test_data)
        self._check_type(out, res)

        # Given float dtype.
        @torch.jit.script
        def func_float(x):
            return torch.arange(x, dtype=torch.float)
        test_data = (torch.tensor(20),)
        out, res = self._test_cvt_to_disc(func_float, test_data)
        self._check_type(out, res)

        # None dtype. Infer dtype from input data.
        @torch.jit.script
        def func_none(x):
            return torch.arange(x)
        # int data
        test_data = (torch.tensor(10, dtype=torch.int),)
        out, res = self._test_cvt_to_disc(func_none, test_data)
        self._check_type(out, res)
        # float data
        test_data = (torch.tensor(10, dtype=torch.float),)
        out, res = self._test_cvt_to_disc(func_none, test_data)
        self._check_type(out, res)


if __name__ == "__main__":
    unittest.main()

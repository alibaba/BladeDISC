import torch
import unittest
import torch_addons

from tests.mlir.testing_utils import DiscTestCase


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

    def _test_binary_ops(self, binary_ops_func):
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

    def _test_func(self, torch_func):
        @torch.jit.script
        def func1(x, y):
            return torch_func(x, y.expand_as(x))

        self._test_binary_ops(func1)

        @torch.jit.script
        def func2(x, y):
            return torch_func(x, y)

        self._test_binary_ops(func2)

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

        self._test_binary_ops(func1)

        @torch.jit.script
        def func2(x, y):
            return torch_func(x, y, alpha=0.1)

        self._test_binary_ops(func2)

    @unittest.skipIf(not torch_addons.version.cuda_available, """Please fix cpu ral first, will raise:
        Context catch an exception: api_func ral_recv_input___pvoid_i32___m2df64 not found
        Context catch an exception: api_func ral_constant_host___pvoid_pvoid___m0df64 not found""")
    def test_binary_type_promotion(self):
        self._test_binary_type_promotion(torch.sub)
        self._test_binary_type_promotion(torch.add)
        self._test_binary_type_promotion(torch.mul)
        self._test_binary_type_promotion(torch.div)
        self._test_binary_type_promotion(torch.rsub)

    def test_arithmetic_func(self):
        self._test_func(torch.sub)
        self._test_func(torch.add)
        self._test_func(torch.mul)
        self._test_func(torch.div)
        self._test_func(torch.rsub)

    def test_arithmetic_func_has_alpha(self):
        self._test_func_has_alpha(torch.sub)
        self._test_func_has_alpha(torch.add)
        self._test_func_has_alpha(torch.rsub)

    def test_scalar_arithmetic_func(self):
        self._test_scalar_func(torch.sub)
        self._test_scalar_func(torch.add)
        self._test_scalar_func(torch.mul)
        self._test_scalar_func(torch.div)
        self._test_rhs_scalar_func(torch.rsub)

    @unittest.skipIf(not torch_addons.version.cuda_available, """disc cpu does not support pow""")
    def test_pow(self):
        self._test_func(torch.pow)
        self._test_scalar_func(torch.pow)

    def _test_cmp_func(self, torch_func):
        @torch.jit.script
        def func(x, y):
            return torch_func(x, 2.0), torch_func(y, 2.0)
        self._test_func(func)

    def test_gt(self):
        self._test_func(torch.gt)
        self._test_cmp_func(torch.gt)

    def test_ge(self):
        self._test_func(torch.ge)
        self._test_cmp_func(torch.ge)

    def test_eq(self):
        self._test_func(torch.eq)
        self._test_cmp_func(torch.eq)

    def test_le(self):
        self._test_func(torch.le)
        self._test_cmp_func(torch.le)

    def test_lt(self):
        self._test_func(torch.lt)
        self._test_cmp_func(torch.lt)

    def test_arange(self):
        @torch.jit.script
        def func1(x):
            return torch.arange(x.size(0), dtype=torch.long)

        test_data = (torch.randn([10, 2, 3, 4], device=self.device),)
        out, res = self._test_cvt_to_disc(func1, test_data)
        self._check_type(out, res)


if __name__ == "__main__":
    unittest.main()

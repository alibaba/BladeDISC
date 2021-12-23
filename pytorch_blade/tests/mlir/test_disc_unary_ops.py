import torch
import unittest
import torch_blade

from tests.mlir.testing_utils import DiscTestCase


class TestDiscUnaryOps(DiscTestCase):
    def _test_unary_ops(self, unary_ops_func):
        x = torch.randn([2, 3, 224, 224]).to(self.device)
        test_data = (x[:, :, :128, :],)
        self._test_cvt_to_disc(unary_ops_func, test_data)

    def test_rsqrt(self):
        @torch.jit.script
        def rsqrt_func(x):
            return x.rsqrt()

        self._test_unary_ops(rsqrt_func)

    def test_exp(self):
        @torch.jit.script
        def exp_func(x):
            return x.exp()

        self._test_unary_ops(exp_func)

    @unittest.skipIf(not torch_blade.version.cuda_available, """disc cpu does not support erf""")
    def test_erf(self):
        @torch.jit.script
        def erf_func(x):
            return x.erf()

        self._test_unary_ops(erf_func)

    def test_neg(self):
        @torch.jit.script
        def neg_func(x):
            return -x

        self._test_unary_ops(neg_func)

    def test_tanh(self):
        @torch.jit.script
        def tanh_func(x):
            return -x

        self._test_unary_ops(tanh_func)

    def test_contiguous(self):
        @torch.jit.script
        def contiguous_func(x):
            return x.contiguous()

        self._test_unary_ops(contiguous_func)

    def test_to_dtype(self):
        @torch.jit.script
        def to_func(x):
            return x.to(torch.int)

        self._test_unary_ops(to_func)


if __name__ == "__main__":
    unittest.main()

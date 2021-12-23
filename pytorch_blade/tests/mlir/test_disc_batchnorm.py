import torch
import unittest

from tests.mlir.testing_utils import DiscTestCase


class TestDiscBatchNorm(DiscTestCase):
    def _test_batchnorm(self, batchnorm_func, inp_test_data=None):
        if inp_test_data is not None:
            test_data = inp_test_data
        else:
            test_data = torch.randn([20, 16, 50, 100], device=self.device)
        if (isinstance(test_data, torch.Tensor)):
            test_data = (test_data.to(self.device),)
        self._test_cvt_to_disc(batchnorm_func, test_data)

    def test_batchnorm1d(self):
        batchnorm = torch.nn.BatchNorm1d(16)
        self._test_batchnorm(batchnorm, torch.randn([20, 16, 60], device=self.device))
        batchnorm = torch.nn.BatchNorm1d(16, affine=False)
        self._test_batchnorm(batchnorm, torch.randn([20, 16, 60], device=self.device))

    def test_batchnorm2d(self):
        batchnorm = torch.nn.BatchNorm2d(16)
        self._test_batchnorm(batchnorm)
        batchnorm = torch.nn.BatchNorm2d(16, affine=False)
        self._test_batchnorm(batchnorm)

    def test_batchnorm3d(self):
        batchnorm = torch.nn.BatchNorm3d(16)
        self._test_batchnorm(batchnorm, torch.randn([20, 16, 60, 50, 100], device=self.device))
        batchnorm = torch.nn.BatchNorm3d(16, affine=False)
        self._test_batchnorm(batchnorm, torch.randn([20, 16, 60, 50, 100], device=self.device))

    def test_functional_batchnorm(self):
        @torch.jit.script
        def batchnorm_func(x, running_mean, running_var, weight, bias):
            out_y = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias)
            return out_y

        channel = 16
        x = torch.ones([20, channel], device=self.device)
        running_mean = torch.randn(channel, device=self.device)
        running_var = torch.randn(channel, device=self.device)
        weight = torch.randn(channel, device=self.device)
        bias = torch.randn(channel, device=self.device)
        self._test_batchnorm(batchnorm_func, (x, running_mean, running_var, weight, bias))

if __name__ == "__main__":
    unittest.main()

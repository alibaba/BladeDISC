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

from tests.disc.testing_base import DiscTestCase


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
        batchnorm = torch.nn.BatchNorm1d(16).eval()
        self._test_batchnorm(batchnorm, torch.randn([20, 16, 60], device=self.device))
        batchnorm = torch.nn.BatchNorm1d(16, affine=False).eval()
        self._test_batchnorm(batchnorm, torch.randn([20, 16, 60], device=self.device))

    def test_batchnorm2d(self):
        batchnorm = torch.nn.BatchNorm2d(16).eval()
        self._test_batchnorm(batchnorm)
        batchnorm = torch.nn.BatchNorm2d(16, affine=False).eval()
        self._test_batchnorm(batchnorm)

    def test_batchnorm3d(self):
        batchnorm = torch.nn.BatchNorm3d(16).eval()
        self._test_batchnorm(batchnorm, torch.randn([20, 16, 60, 50, 100], device=self.device))
        batchnorm = torch.nn.BatchNorm3d(16, affine=False).eval()
        self._test_batchnorm(batchnorm, torch.randn([20, 16, 60, 50, 100], device=self.device))

    def test_functional_batchnorm(self):
        @torch.jit.script
        def batchnorm_func(x, running_mean, running_var, weight, bias):
            out_y = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, training=False)
            return out_y

        channel = 16
        x = torch.ones([20, channel], device=self.device)
        running_mean = torch.randn(channel, device=self.device)
        running_var = torch.randn(channel, device=self.device)
        weight = torch.randn(channel, device=self.device)
        bias = torch.randn(channel, device=self.device)
        with torch.no_grad():
            self._test_batchnorm(batchnorm_func, (x, running_mean, running_var, weight, bias))

if __name__ == "__main__":
    unittest.main()

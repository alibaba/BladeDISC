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
import pytest
import torch
import unittest
from unittest import skipIf

from tests.disc.testing_base import DiscTestCase, skipTorchLE

class TestDiscUnaryOps(DiscTestCase):
    def _test_unary_ops(self, unary_ops_func, test_data=None):
        x = torch.randn([2, 3, 224, 224]).to(self.device)
        if test_data is None:
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

    def test_bitwise_not(self):
        @torch.jit.script
        def not_func(x):
            return torch.bitwise_not(x)

        # Please fix disc compilation error of the following test:
        # https://bladedisc-ci.oss-cn-hongkong.aliyuncs.com/download/debug/dump_dir.bitwise_not.tar.gz
        # x = torch.randint(0, 2, [2, 3, 224, 224],
        #                   dtype=torch.int32, device=self.device)
        # self._test_unary_ops(not_func, (x,))

        y = torch.randint(0, 2, [6, 224], dtype=torch.bool, device=self.device)
        self._test_unary_ops(not_func, (y,))

    def test_tanh(self):
        @torch.jit.script
        def tanh_func(x):
            return torch.tanh(x)

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

    def test_type_as(self):
        @torch.jit.script
        def type_as(x, y):
            return x.type_as(y)

        x = torch.randn([2, 3, 224, 224], dtype=torch.float32, device=self.device)
        y = torch.randn([6, 224], dtype=torch.half, device=self.device)
        test_data = (x, y)
        self._test_cvt_to_disc(type_as, test_data)

    # test aten::to(prim::dtype(x))
    @skipTorchLE("1.8.1")
    def test_type(self):
        @torch.jit.script
        def tensor_type(x):
            return torch.softmax(x.float(), dim=-1).type(x.dtype)

        x = torch.randn([2, 3, 224, 224], dtype=torch.half, device=self.device)
        test_data = (x, )
        self._test_cvt_to_disc(tensor_type, test_data=test_data)

if __name__ == "__main__":
    unittest.main()

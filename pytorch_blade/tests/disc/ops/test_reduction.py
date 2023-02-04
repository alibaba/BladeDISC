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

from tests.disc.testing_base import DiscTestCase, skipTorchGE
from torch_blade import utils


class TestDiscReduction(DiscTestCase):
    def _test_reduction(self, reduce_func, dtype=None):
        dtype = torch.float if dtype is None else dtype
        if dtype in {torch.int32, torch.int64}:
            x = torch.randint(-256, 256, [2, 3, 224, 224], dtype=dtype).to(self.device)
        else:
            x = torch.randn([2, 3, 224, 224], dtype=dtype).to(self.device)

        test_data = (x,)
        annotation = ([-1, -1, -1, -1], dtype)
        self._test_disc(reduce_func, [annotation, annotation], test_data, rtol=1e-3, atol=1e-2)

    def test_cvt_to_disc_sum_f64(self):
        @torch.jit.script
        def sum_func(x):
            return torch.sum(x)

        self._test_reduction(sum_func, dtype=torch.float64)

    def test_cvt_to_disc_sum_f32(self):
        @torch.jit.script
        def sum_func(x):
            return torch.sum(x)

        self._test_reduction(sum_func, dtype=torch.float32)

    def test_cvt_to_disc_sum_i32(self):
        @torch.jit.script
        def sum_func(x):
            return torch.sum(x, dtype=torch.int32)

        self._test_reduction(sum_func, dtype=torch.int32)

    def test_cvt_to_disc_sum_dtype_f32(self):
        @torch.jit.script
        def sum_func(x):
            return torch.sum(x, dtype=torch.float32)

        self._test_reduction(sum_func)

    def test_cvt_to_disc_sum_dtype_f64(self):
        @torch.jit.script
        def sum_func(x):
            return torch.sum(x, dtype=torch.float64)

        self._test_reduction(sum_func)

    @skipTorchGE("1.12.0")
    def test_cvt_to_disc_sum_list_f32(self):
        @torch.jit.script
        def sum_func(x):
            return torch.sum(x, [1, 2, 3])

        self._test_reduction(sum_func)

    def test_cvt_to_disc_sum_list_keepdim(self):
        @torch.jit.script
        def sum_func(x):
            return torch.sum(x, [3, 1], keepdim=True)

        self._test_reduction(sum_func)

    def test_cvt_to_disc_mean_list_keepdim(self):
        @torch.jit.script
        def sum_func(x):
            return torch.mean(x, [3, 1], keepdim=True)

        self._test_reduction(sum_func)

    def test_cvt_to_disc_mean_list(self):
        @torch.jit.script
        def sum_func(x):
            return torch.mean(x, [3, 0, 1])

        self._test_reduction(sum_func)

    @unittest.skipIf(
        utils.torch_version_number() >= utils.parse_version("1.10.0"),
        "mean(): input dtype should be either floating point or complex dtypes.",
    )
    def test_cvt_to_disc_mean_dtype_i32(self):
        @torch.jit.script
        def sum_func(x):
            return torch.mean(x, dtype=torch.float32)

        self._test_reduction(sum_func, dtype=torch.int32)

    def test_cvt_to_disc_mean_dtype_f32(self):
        @torch.jit.script
        def sum_func(x):
            return torch.mean(x, dtype=torch.float32)

        self._test_reduction(sum_func, dtype=torch.float32)

    def test_cvt_to_disc_mean_f32(self):
        @torch.jit.script
        def sum_func(x):
            return torch.mean(x, [0, 1, 2, 3])

        self._test_reduction(sum_func)

    def test_cvt_to_disc_multi_reduce(self):
        @torch.jit.script
        def sum_func(x):
            a = torch.mean(x, [0, 2, 3])
            b = torch.sum(x, [0, 2, 3])
            return a * b

        def test_reduction(reduce_func, dtype=None):
            annotation = ([-1, -1, -1, -1], dtype)
            x = torch.randn([2, 3, 25, 25], dtype=dtype, device=self.device).to(self.device) * 0.1
            self._test_disc(reduce_func, [annotation, annotation], (x, ), rtol=1e-3, atol=1e-2)
            x = torch.randn([2, 3, 254, 254], dtype=dtype, device=self.device).to(self.device) * 0.01
            self._test_disc(reduce_func, [annotation, annotation], (x, ), rtol=1e-3, atol=1e-2)

        if self.device == torch.device('cuda'):
            test_reduction(sum_func, torch.half)
        test_reduction(sum_func, torch.float)


if __name__ == "__main__":
    unittest.main()

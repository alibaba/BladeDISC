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
from unittest import skip

from tests.mlir.testing_utils import DiscTestCase


class TestDiscReduction(DiscTestCase):
    def _test_reduction(self, reduce_func, dtype=None):
        if dtype in {torch.int32, torch.int64}:
            x = torch.randint(-256, 256, [2, 3, 224, 224], dtype=dtype).to(self.device)
        else:
            x = torch.randn([2, 3, 224, 224], dtype=dtype).to(self.device)

        test_data = (x,)
        self._test_cvt_to_disc(reduce_func, test_data, rtol=1e-3, atol=3e-3)

    @skip("need aicompiler support f64")
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

    @skip("fix runtime core dump first")
    def test_cvt_to_disc_sum_i32(self):
        @torch.jit.script
        def sum_func(x):
            return torch.sum(x)

        self._test_reduction(sum_func, dtype=torch.int32)

    def test_cvt_to_disc_sum_dtype_f32(self):
        @torch.jit.script
        def sum_func(x):
            return torch.sum(x, dtype=torch.float32)

        self._test_reduction(sum_func)

    @skip("need aicompiler support f64")
    def test_cvt_to_disc_sum_dtype_f64(self):
        @torch.jit.script
        def sum_func(x):
            return torch.sum(x, dtype=torch.float64)

        self._test_reduction(sum_func)

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


if __name__ == "__main__":
    unittest.main()

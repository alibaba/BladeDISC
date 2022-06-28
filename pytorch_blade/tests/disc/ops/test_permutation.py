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
from typing import List
import unittest

from torch_blade import tools
from tests.disc.testing_base import DiscTestCase, isTorchMlirEnable


class TestDiscPermutation(DiscTestCase):
    def _test_permute(self, reshape_func, dtype=None, x=None):
        dtype=torch.float if dtype is None else dtype
        x = torch.randn([2, 3, 224, 224], dtype=dtype, device=self.device) if x is None else x
        test_data = (x,)
        if len(x.shape) > 0:
            annotations = [([-1] * len(x.shape), dtype)]
        else:
            annotations = []
        self._test_disc(reshape_func, annotations, test_data)

    def test_transpose(self):
        @torch.jit.script
        def transpose(x):
            return x.transpose(1, 3)

        self._test_permute(transpose)

        @torch.jit.script
        def transpose_neg_index(x):
            return x.transpose(-2, -3)

        self._test_permute(transpose_neg_index)

    def test_aten_t(self):
        @torch.jit.script
        def transpose(x):
            return x.t()

        if not isTorchMlirEnable():
            # Failed to materialize conversion for result #0 of operation
            # 'torch.aten.t' that remained live after conversion
            self._test_permute(transpose, x=torch.tensor(5, device=self.device))
            self._test_permute(transpose, x=torch.randn([], device=self.device))
            self._test_permute(transpose, x=torch.randn([5], device=self.device))
        self._test_permute(transpose, x=torch.randn([3, 4], device=self.device))

    def test_permute(self):
        class TestModel(torch.nn.Module):
            permute_dims: List[int]

            def __init__(self, permute_dims):
                super(TestModel, self).__init__()
                self.permute_dims = permute_dims

            def forward(self, x):
                return x.permute(self.permute_dims)

        permute = TestModel([])
        self._test_permute(permute, x=torch.tensor(5, device=self.device))
        permute = TestModel([])
        self._test_permute(permute, x=torch.randn([], device=self.device))
        permute = TestModel([0])
        self._test_permute(permute, x=torch.randn([5], device=self.device))
        permute = TestModel([1, 0])
        self._test_permute(permute, x=torch.randn([3, 4], device=self.device))
        permute = TestModel([1, -1, 0])
        self._test_permute(permute, x=torch.randn([2, 3, 4], device=self.device))

if __name__ == "__main__":
    unittest.main()

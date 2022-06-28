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
from tests.disc.testing_base import DiscTestCase, skipIfEnableTorchMlir, isTorchMlirEnable


class TestDiscMemOps(DiscTestCase):
    def _test_mem_ops(self, reshape_func, dtype=None, x=None):
        dtype=torch.float if dtype is None else dtype
        x = torch.randn([2, 3, 224, 224], dtype=dtype, device=self.device) if x is None else x
        test_data = (x,)
        if len(x.shape) > 0:
            annotations = [([-1] * len(x.shape), dtype)]
        else:
            annotations = []
        self._test_disc(reshape_func, annotations, test_data)

    def test_roll(self):

        @torch.jit.script
        def func(x):
            z = torch.roll(x, shifts=(3, -9), dims=(1, 0))
            return z

        x = torch.randn([4, 64, 256], device=self.device)
        self._test_mem_ops(func, x=x)

    def test_index_select(self):

        @torch.jit.script
        def func(x):
            indices = torch.tensor([0, 2], device=x.device)
            y = torch.index_select(x, 0, indices)
            return y

        x = torch.randn([3, 4], device=self.device)
        self._test_mem_ops(func, x=x)

    def test_flip(self):

        @torch.jit.script
        def func(x):
            y = torch.flip(x, [0, 1])
            return y

        x = torch.arange(8).view(2, 2, 2).to(self.device)
        self._test_mem_ops(func, x=x)


if __name__ == "__main__":
    unittest.main()

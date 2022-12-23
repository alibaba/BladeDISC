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


class TestDiscSlices(DiscTestCase):
    def _test_slice(self, reshape_func, dtype=None, x=None):
        dtype=torch.float if dtype is None else dtype
        x = torch.randn([2, 3, 224, 224], dtype=dtype, device=self.device) if x is None else x
        test_data = (x,)
        if len(x.shape) > 0:
            annotations = [([-1] * len(x.shape), dtype)]
        else:
            annotations = []
        self._test_disc(reshape_func, annotations, test_data)

    def test_slices(self):
        @torch.jit.script
        def slice_func(x):
            return x[-50:-1]

        self._test_slice(slice_func)
        if not isTorchMlirEnable():
            self._test_slice(slice_func, x=torch.randn([2, 0, 0, 224]))

        @torch.jit.script
        def slice_func(x):
            return x[-50:-1, 50:, :-50]

        # TorchMhlo(disc): expected result type with stride = 1 instead of 0 in dim = 0
        if not isTorchMlirEnable():
            self._test_slice(slice_func)
            self._test_slice(slice_func, x=torch.randn([2, 0, 0, 224]))

        @torch.jit.script
        def slice_func(x):
            return x[-50:2008, :-2, 4:, :]

        if not isTorchMlirEnable():
            self._test_slice(slice_func)
            self._test_slice(slice_func, x=torch.randn([2, 0, 0, 224]))

    def test_dyn_slices(self):
        @torch.jit.script
        def dyn_slice_func(x, y):
            d = y.size(1)
            return x[:,0:d]

        x = torch.randn([224, 224], device=self.device)
        y = torch.randn([6, 112], device=self.device)
        test_data = (x, y)
        dtype = torch.float
        annotations = [([-1, -1], dtype), ([-1, -1], dtype)]
        self._test_disc(dyn_slice_func, annotations, test_data)

    def test_select(self):
        @torch.jit.script
        def select_func(x):
            return x[0]

        self._test_slice(select_func)

        @torch.jit.script
        def select_func(x):
            return x.select(-1, -1)

        self._test_slice(select_func)

        @torch.jit.script
        def select_func(x):
            d = x.size(-2)
            return x.select(-2, d - 1)

        if not isTorchMlirEnable():
            self._test_slice(select_func)

    def test_cat_slice_select(self):
        x = torch.randn([4, 64, 256], device=self.device)
        y = torch.randn([4, 1, 256], device=self.device)
        test_data = (x, y)

        @torch.jit.script
        def func(x, y):
            z = torch.cat([x, y], dim=1)
            return z[:, -1]

        dtype = torch.float
        annotations = [([4, -1, 256], dtype), ([4, -1, 256], dtype)]
        self._test_cvt_to_disc(func, test_data, annotations)

    def test_narrow(self):

        @torch.jit.script
        def func(x):
            return torch.narrow(x, 1, 32, 16)

        x = torch.randn([4, 64, 12], device=self.device)
        self._test_slice(func, x=x)
        x = torch.randn([4, 48, 12], device=self.device)
        self._test_slice(func, x=x)

    def test_unbind(self):
        x = torch.randn([4, 64, 256], device=self.device)
        y = torch.randn([1, 4, 256], device=self.device)
        test_data = (x, y)

        @torch.jit.script
        def func(x, y):
            a0, b0, c0, d0 = torch.unbind(x, dim=0)
            a1, b1, c1, d1 = torch.unbind(y, dim=1)
            a = a0 + a1
            b = b0 * b1
            c = c0 - c1
            d = d0 / d1
            return a + b + c + d

        dtype = torch.float
        annotations = [([4, -1, -1], dtype), ([1, 4, -1], dtype)]

        with tools.trust_tracing_shape():
            self._test_cvt_to_disc(func, test_data, annotations)

    def test_chunk(self):

        @torch.jit.script
        def func(x):
            z1, z2, z3, z4, z5, z6 = torch.chunk(x, 6, -1)
            return z1, z2, z3, z4, z5, z6

        x = torch.randn([4, 64, 11], device=self.device)
        self._test_slice(func, x=x)

        x = torch.randn([4, 64, 12], device=self.device)
        self._test_slice(func, x=x)

    def test_split(self):

        @torch.jit.script
        def func(x):
            z1, z2, z3, z4, z5, z6 = torch.split(x, 2, -1)
            return z1, z2, z3, z4, z5, z6

        x = torch.randn([4, 64, 11], device=self.device)
        annotations = [([-1, -1, 11], torch.float)]
        self._test_disc(func, annotations, (x, ))
        annotations = [([-1, -1, -1], torch.float)]
        self._test_disc(func, annotations, (x, ))

        x = torch.randn([4, 64, 12], device=self.device)
        annotations = [([4, -1, 12], torch.float)]
        self._test_disc(func, annotations, (x, ))
        annotations = [([4, -1, -1], torch.float)]
        self._test_disc(func, annotations, (x, ))

if __name__ == "__main__":
    unittest.main()

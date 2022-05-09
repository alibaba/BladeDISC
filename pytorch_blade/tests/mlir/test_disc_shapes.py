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
from tests.mlir.testing_utils import DiscTestCase


class TestDiscShapes(DiscTestCase):
    def _test_reshape(self, reshape_func, dtype=None, x=None):
        x = torch.randn([2, 3, 224, 224], dtype=dtype, device=self.device) if x is None else x
        test_data = (x,)
        self._test_cvt_to_disc(reshape_func, test_data)

    def test_dyn_reshape(self):
        @torch.jit.script
        def reshape_as(x, y):
            return x.reshape(y.size())

        x = torch.randn([2, 3, 224, 224], device=self.device)
        y = torch.randn([6, 224, 224], device=self.device)
        test_data = (x, y)
        self._test_cvt_to_disc(reshape_as, test_data)

    def test_view_as(self):
        @torch.jit.script
        def view_as(x, y):
            return x.view_as(y)

        x = torch.randn([2, 3, 224, 224], device=self.device)
        y = torch.randn([6, 224, 224], device=self.device)
        test_data = (x, y)
        self._test_cvt_to_disc(view_as, test_data)

    def test_reshape(self):
        @torch.jit.script
        def static_reshape(x):
            return x.reshape([2 * 3 * 224, 224])

        self._test_reshape(static_reshape)

    def test_view(self):
        @torch.jit.script
        def static_view(x):
            return x.view([2 * 3 * 224, 224])

        self._test_reshape(static_view)

    def test_scalar_view(self):
        @torch.jit.script
        def scalar_view(x):
            shape: List[int] = []
            return x.view(shape)

        self._test_reshape(scalar_view, x=torch.randn([1]))

    def test_dynamic_view(self):
        @torch.jit.script
        def dynamic_view(x):
            shape = [-1, 120, 4, 64]
            return x.view(shape)

        self._test_reshape(dynamic_view, x=torch.ones(1, 4, 120, 256))

    def test_collapse_reshape(self):
        @torch.jit.script
        def dynamic_reshape(x):
            return x.reshape([-1, 224])

        self._test_reshape(dynamic_reshape)

        def dynamic_reshape_1(x):
            return x.reshape([-1])

        self._test_reshape(dynamic_reshape)

    def test_transpose(self):
        @torch.jit.script
        def transpose(x):
            return x.transpose(1, 3)

        self._test_reshape(transpose)

        @torch.jit.script
        def transpose_neg_index(x):
            return x.transpose(-2, -3)

        self._test_reshape(transpose_neg_index)

    def test_aten_t(self):
        @torch.jit.script
        def transpose(x):
            return x.t()

        self._test_reshape(transpose, x=torch.tensor(5, device=self.device))
        self._test_reshape(transpose, x=torch.randn([], device=self.device))
        self._test_reshape(transpose, x=torch.randn([5], device=self.device))
        self._test_reshape(transpose, x=torch.randn([3, 4], device=self.device))

    def test_permute(self):
        class TestModel(torch.nn.Module):
            permute_dims: List[int]

            def __init__(self, permute_dims):
                super(TestModel, self).__init__()
                self.permute_dims = permute_dims

            def forward(self, x):
                return x.permute(self.permute_dims)

        permute = TestModel([])
        self._test_reshape(permute, x=torch.tensor(5, device=self.device))
        permute = TestModel([])
        self._test_reshape(permute, x=torch.randn([], device=self.device))
        permute = TestModel([0])
        self._test_reshape(permute, x=torch.randn([5], device=self.device))
        permute = TestModel([1, 0])
        self._test_reshape(permute, x=torch.randn([3, 4], device=self.device))
        permute = TestModel([1, -1, 0])
        self._test_reshape(permute, x=torch.randn([2, 3, 4], device=self.device))


    def test_unsqueeze(self):

        @torch.jit.script
        def unsqueeze_0(x):
            return x.unsqueeze(0)

        self._test_reshape(unsqueeze_0)

        @torch.jit.script
        def unsqueeze_1(x):
            return x.unsqueeze(-2)

        self._test_reshape(unsqueeze_1)

        @torch.jit.script
        def unsqueeze_2(x):
            return x.unsqueeze(1)

        self._test_reshape(unsqueeze_2)

    def test_squeeze(self):
        x = torch.zeros(2, 1, 2, 1, 2, device=self.device)

        @torch.jit.script
        def squeeze_0(x):
            return x.squeeze(0)

        self._test_reshape(squeeze_0, x=x)

        @torch.jit.script
        def squeeze_1(x):
            return x.squeeze(-2)

        self._test_reshape(squeeze_1, x=x)

        @torch.jit.script
        def squeeze_2(x):
            return x.squeeze(1)

        self._test_reshape(squeeze_2, x=x)

        @torch.jit.script
        def squeeze_3(x):
            return x.squeeze()

        self._test_reshape(squeeze_3, x=x)

    def test_slices(self):
        @torch.jit.script
        def slice_func(x):
            return x[-50:-1]

        self._test_reshape(slice_func)
        self._test_reshape(slice_func, x=torch.randn([2, 0, 0, 224]))

        @torch.jit.script
        def slice_func(x):
            return x[-50:-1, 50:, :-50]

        self._test_reshape(slice_func)
        self._test_reshape(slice_func, x=torch.randn([2, 0, 0, 224]))

        @torch.jit.script
        def slice_func(x):
            return x[-50:2008, :-2, 4:, :]

        self._test_reshape(slice_func)
        self._test_reshape(slice_func, x=torch.randn([2, 0, 0, 224]))

    def test_dyn_slices(self):
        @torch.jit.script
        def dyn_slice_func(x, y):
            d = y.size(1)
            return x[:,0:d]

        x = torch.randn([224, 224], device=self.device)
        y = torch.randn([6, 112], device=self.device)
        test_data = (x, y)
        self._test_cvt_to_disc(dyn_slice_func, test_data)

    def test_flatten(self):
        @torch.jit.script
        def basic_test_0(x):
            return torch.flatten(x)

        @torch.jit.script
        def basic_test_1(x):
            return torch.flatten(x, 1)

        @torch.jit.script
        def basic_test_2(x):
            return torch.flatten(x, 2, 3)

        x = torch.randn([2, 3, 224, 224], device=self.device)
        test_data = (x,)
        self._test_cvt_to_disc(basic_test_0, test_data)
        self._test_cvt_to_disc(basic_test_1, test_data)
        self._test_cvt_to_disc(basic_test_2, test_data)

        @torch.jit.script
        def test_rank_0_input(x):
            return torch.flatten(x)

        x = torch.randn([], device=self.device)
        test_data = (x,)
        self._test_cvt_to_disc(test_rank_0_input, test_data)

    def test_size(self):
        @torch.jit.script
        def size_func(x):
            return x + x.size()[1]

        self._test_reshape(size_func)

        @torch.jit.script
        def size_func(x):
            return x + x.size(1)

        self._test_reshape(size_func)

    def test_select(self):
        @torch.jit.script
        def select_func(x):
            return x[0]

        self._test_reshape(select_func)

        @torch.jit.script
        def select_func(x):
            return x.select(-1, -1)

        self._test_reshape(select_func)

        @torch.jit.script
        def select_func(x):
            d = x.size(-2)
            return x.select(-2, d - 1)

        self._test_reshape(select_func)

    def test_cat_slice_select(self):
        x = torch.randn([4, 64, 256], device=self.device)
        y = torch.randn([4, 1, 256], device=self.device)
        test_data = (x, y)

        @torch.jit.script
        def func(x, y):
            z = torch.cat([x, y], dim=1)
            return z[:, -1]

        self._test_cvt_to_disc(func, test_data)

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

        with tools.trust_tracing_shape():
            self._test_cvt_to_disc(func, test_data)

    def test_roll(self):
        x = torch.randn([4, 64, 256], device=self.device)
        test_data = (x, )

        @torch.jit.script
        def func(x):
            z = torch.roll(x, shifts=(3, -9), dims=(1, 0))
            return z

        self._test_cvt_to_disc(func, test_data)

    def test_index_select(self):
        x = torch.randn([3, 4], device=self.device)
        test_data = (x, )

        @torch.jit.script
        def func(x):
            indices = torch.tensor([0, 2], device=x.device)
            y = torch.index_select(x, 0, indices)
            return y

        self._test_cvt_to_disc(func, test_data)

    def test_flip(self):
        x = torch.arange(8).view(2, 2, 2).to(self.device)
        test_data = (x, )

        @torch.jit.script
        def func(x):
            y = torch.flip(x, [0, 1])
            return y

        self._test_cvt_to_disc(func, test_data)

    def test_chunk(self):

        @torch.jit.script
        def func(x):
            z1, z2, z3, z4, z5, z6 = torch.chunk(x, 6, -1)
            return z1, z2, z3, z4, z5, z6

        x = torch.randn([4, 64, 11], device=self.device)
        test_data = (x, )
        self._test_cvt_to_disc(func, test_data)

        x = torch.randn([4, 64, 12], device=self.device)
        test_data = (x, )
        self._test_cvt_to_disc(func, test_data)

if __name__ == "__main__":
    unittest.main()

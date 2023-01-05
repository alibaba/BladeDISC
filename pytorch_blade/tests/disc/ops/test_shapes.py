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
from tests.disc.testing_base import DiscTestCase, skipIfEnableTorchMlir

class TestDiscShapes(DiscTestCase):
    def _test_reshape(self, reshape_func, dtype=None, x=None):
        dtype = torch.float if dtype is None else dtype
        x = torch.randn([2, 3, 224, 224], dtype=dtype, device=self.device) if x is None else x
        test_data = (x,)
        annotation = ([-1] * len(x.shape), dtype)
        self._test_disc(reshape_func, [annotation], test_data)

    def test_dyn_reshape(self):
        @torch.jit.script
        def reshape_as(x, y):
            return x.reshape(y.size())

        x = torch.randn([2, 3, 224, 224], device=self.device)
        y = torch.randn([6, 224, 224], device=self.device)
        test_data = (x, y)
        dtype = torch.float
        annotations = [([-1,-1,-1,-1], dtype), ([-1,-1,-1], dtype)]
        self._test_disc(reshape_as, annotations, test_data)

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
        annotation = ([2, 1, 2, 1, 2], torch.float)

        @torch.jit.script
        def squeeze_0(x):
            return x.squeeze(0)

        self._test_disc(squeeze_0, [annotation], (x,))

        @torch.jit.script
        def squeeze_1(x):
            return x.squeeze(-2)

        self._test_disc(squeeze_1, [annotation], (x,))

        @torch.jit.script
        def squeeze_2(x):
            return x.squeeze(1)

        self._test_disc(squeeze_2, [annotation], (x,))

        @torch.jit.script
        def squeeze_3(x):
            return x.squeeze()

        self._test_disc(squeeze_3, [annotation], (x,))

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

        self._test_reshape(basic_test_0)
        self._test_reshape(basic_test_1)
        self._test_reshape(basic_test_2)

        @torch.jit.script
        def test_rank_0_input(x):
            return torch.flatten(x)

        x = torch.randn([], device=self.device)
        self._test_reshape(test_rank_0_input, x=x)

    def test_size(self):
        @torch.jit.script
        def size_func(x):
            return x + x.size()[1]

        self._test_reshape(size_func)

        @torch.jit.script
        def size_func(x):
            return x + x.size(1)

        self._test_reshape(size_func)

    def test_extract_op(self):

        class sample(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim
            def forward(self, x):
                a, b ,c, d = x.shape
                dim = self.dim
                # convert to torch.aten.Int.Tensor %6 : !torch.tensor<[],f32> -> !torch.int in torch-mlir
                x = x.reshape(a, b, c, dim, d//dim)
                return x
        s = sample(dim=2)
        self._test_reshape(s)

if __name__ == "__main__":
    unittest.main()

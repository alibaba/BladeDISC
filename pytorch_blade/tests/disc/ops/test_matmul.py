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

from tests.disc.testing_base import DiscTestCase, skipIfEnableTorchMlir
from unittest import skipIf

class MatMul(torch.nn.Module):
    def __init__(self, device=torch.device('cuda')):
        super().__init__()
        self.weight = torch.ones(256, 256, device=device)

    def forward(self, y):
        out_y = torch.matmul(y, self.weight)
        return out_y


class Linear(torch.nn.Module):
    def __init__(self, device=torch.device('cuda'), dtype=torch.float):
        super().__init__()
        self.weight = torch.ones(256, 256, device=device, dtype=dtype)
        self.bias = torch.ones(120, 256, device=device, dtype=dtype)

    def forward(self, x, y):
        out_y = torch.matmul(y, self.weight)
        out_b = out_y + self.bias
        return x + out_b


class TestDiscMatMul(DiscTestCase):
    def _test_linear(self, dtype):
        x = torch.randn(4, 120, 256, dtype=dtype).to(self.device)
        y = torch.randn(4, 120, 256, dtype=dtype).to(self.device)
        test_data = (x, y)
        annotation = ([-1, -1, -1], dtype)
        linear = Linear(self.device, dtype).eval()
        self._test_disc(linear, [annotation, annotation], test_data, atol=5e-2, rtol=1e-3)
        linear = torch.nn.Linear(256, 256).to(self.device).to(dtype)
        annotation = ([-1, 120, 256], dtype)
        self._test_disc(linear, [annotation],  (x,), atol=5e-2, rtol=1e-3)

    def test_linear(self):
        self._test_linear(torch.float)

    @skipIf(not torch.cuda.is_available(), "only test half on cuda")
    def test_linear_half(self):
        self._test_linear(torch.half)

    def test_matmul_module(self):
        y = torch.randn(4, 120, 256).to(self.device)
        annotation = ([-1, -1, 256], torch.float)
        matmul = MatMul(self.device).eval()
        self._test_disc(matmul, [annotation], (y,))
        y = torch.randn(120, 256).to(self.device)
        annotation = ([-1, 256], torch.float)
        matmul = MatMul(self.device).eval()
        self._test_disc(matmul, [annotation], (y,))

    def test_matmul(self):
        @torch.jit.script
        def matmul(x, y):
            return torch.matmul(x, y)

        x = torch.randn(4, 120, 256).to(self.device)
        y = torch.randn(4, 256, 120).to(self.device)
        self._test_cvt_to_disc(matmul, (x, y))
        self._test_cvt_to_disc(matmul, (y, x))

        x = torch.randn(4, 120, 256).to(self.device)
        y = torch.randn(256, 120).to(self.device)
        self._test_cvt_to_disc(matmul, (x, y))
        self._test_cvt_to_disc(matmul, (y, x))

        x = torch.randn(1, 256, 256).to(self.device)
        y = torch.randn(256).to(self.device)
        self._test_cvt_to_disc(matmul, (x, y))
        self._test_cvt_to_disc(matmul, (y, x))

        x = torch.randn(120, 256).to(self.device)
        y = torch.randn(256, 120).to(self.device)
        self._test_cvt_to_disc(matmul, (x, y))

        x = torch.randn(256, 256).to(self.device)
        y = torch.randn(256).to(self.device)
        self._test_cvt_to_disc(matmul, (x, y))
        self._test_cvt_to_disc(matmul, (y, x))

        x = torch.randn(256).to(self.device)
        y = torch.randn(256).to(self.device)
        self._test_cvt_to_disc(matmul, (x, y))

        x = torch.randn(256).to(self.device)
        y = torch.randn(256, 1).to(self.device)
        self._test_cvt_to_disc(matmul, (x, y))


    def test_bmm(self):
        @torch.jit.script
        def bmm(x, y):
            return torch.bmm(x, y)

        input = torch.randn(10, 3, 4).to(self.device)
        mat2 = torch.randn(10, 4, 5).to(self.device)
        self._test_cvt_to_disc(bmm, (input, mat2))

    def test_addmm(self):
        @torch.jit.script
        def addmm(M, mat1, mat2):
            return torch.addmm(M, mat1, mat2, beta=0.1, alpha=2.0)

        M = torch.randn(2, 3).to(self.device)
        mat1 = torch.randn(2, 3).to(self.device)
        mat2 = torch.randn(3, 3).to(self.device)
        self._test_cvt_to_disc(addmm, (M, mat1, mat2))

        @torch.jit.script
        def addmm(M, mat1, mat2):
            return torch.addmm(M, mat1, mat2, beta=0.2)

        self._test_cvt_to_disc(addmm, (M, mat1, mat2))

        @torch.jit.script
        def addmm(M, mat1, mat2):
            return torch.addmm(M, mat1, mat2)

        self._test_cvt_to_disc(addmm, (M, mat1, mat2))

    @skipIfEnableTorchMlir()
    def test_einsum(self):
        @torch.jit.script
        def einsum_0(x, y):
            return torch.einsum("ijbn,jbnd->ibnd", [x, y])

        @torch.jit.script
        def einsum_1(x, y):
            return torch.einsum("ibnd,jnd->ijbn", [x, y])

        @torch.jit.script
        def einsum_2(x, y):
            return torch.einsum("ibnd,jbnd->ijbn", [x, y])

        @torch.jit.script
        def einsum_3(x, y):
            return torch.einsum("ibn d,jbnd-> ijbn", [x, y])

        @torch.jit.script
        def einsum_4(x, y):
            return torch.einsum("ij,jk", [x, y])

        i = 4
        j = 8
        b = 16
        n = 20
        d = 32

        # "ijbn,jbnd->ibnd"
        x = torch.randn(i, j, b, n).to(self.device)
        y = torch.randn(j, b, n, d).to(self.device)
        self._test_cvt_to_disc(einsum_0, (x, y))

        # "ibnd,jnd->ijbn"
        x = torch.randn(i, b, n, d).to(self.device)
        y = torch.randn(j, n, d).to(self.device)
        self._test_cvt_to_disc(einsum_1, (x, y))

        # "ibnd,jbnd->ijbn"
        x = torch.randn(i, b, n, d).to(self.device)
        y = torch.randn(j, b, n, d).to(self.device)
        self._test_cvt_to_disc(einsum_2, (x, y))

        # "ibnd,jbnd->ijbn" with blank char
        x = torch.randn(i, b, n, d).to(self.device)
        y = torch.randn(j, b, n, d).to(self.device)
        self._test_cvt_to_disc(einsum_3, (x, y))

        # "ij,jk" with no explicit result
        x = torch.randn(i, j).to(self.device)
        y = torch.randn(j, d).to(self.device)
        self._test_cvt_to_disc(einsum_4, (x, y))

if __name__ == "__main__":
    unittest.main()

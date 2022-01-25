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

from torch_blade.version import cuda_available
from tests.mlir.testing_utils import DiscTestCase

class MatMul(torch.nn.Module):
    def __init__(self, device=torch.device('cuda')):
        super().__init__()
        self.weight = torch.ones(256, 256, device=device)

    def forward(self, y):
        out_y = torch.matmul(y, self.weight)
        return out_y


class Linear(torch.nn.Module):
    def __init__(self, device=torch.device('cuda')):
        super().__init__()
        self.weight = torch.ones(256, 256, device=device)
        self.bias = torch.ones(120, 256, device=device)

    def forward(self, x, y):
        out_y = torch.matmul(y, self.weight)
        out_b = out_y + self.bias
        return x + out_b


class TestDiscMatMul(DiscTestCase):
    def test_linear(self):
        x = torch.randn(4, 120, 256).to(self.device)
        y = torch.randn(4, 120, 256).to(self.device)
        test_data = (x, y)
        linear = Linear(self.device).eval()
        self._test_cvt_to_disc(linear, test_data)

        linear = torch.nn.Linear(256, 256).to(self.device)
        self._test_cvt_to_disc(linear, (x,))

    def test_matmul_module(self):
        y = torch.randn(4, 120, 256).to(self.device)
        matmul = MatMul(self.device).eval()
        self._test_cvt_to_disc(matmul, (y,))

        y = torch.randn(120, 256).to(self.device)
        matmul = MatMul(self.device).eval()
        self._test_cvt_to_disc(matmul, (y,))

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


if __name__ == "__main__":
    unittest.main()

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

from tests.mlir.testing_utils import DiscTestCase


class TestDiscActivation(DiscTestCase):
    def _test_activation(self, activation_func):
        x = torch.randn([2, 4, 16, 16], device=self.device)
        test_data = (x,)
        self._test_cvt_to_disc(activation_func, test_data)

    def test_relu(self):
        relu = torch.nn.ReLU()
        self._test_activation(relu)

        @torch.jit.script
        def relu_func(x):
            return torch.nn.functional.relu(x)

        self._test_activation(relu_func)

    def test_leaky_relu(self):
        leaky_relu = torch.nn.LeakyReLU()
        self._test_activation(leaky_relu)

        @torch.jit.script
        def leaky_relu_func(x):
            return torch.nn.functional.leaky_relu(x)

        self._test_activation(leaky_relu_func)

    def test_sigmoid(self):
        sigmoid_func = torch.nn.Sigmoid()
        self._test_activation(sigmoid_func)

    def test_glu(self):
        @torch.jit.script
        def glu_func(x):
            return torch.nn.functional.glu(x)

        self._test_activation(glu_func)

    def test_gelu(self):
        @torch.jit.script
        def gelu_func(x):
            return torch.nn.functional.gelu(x)
        self._test_activation(gelu_func)

if __name__ == "__main__":
    unittest.main()

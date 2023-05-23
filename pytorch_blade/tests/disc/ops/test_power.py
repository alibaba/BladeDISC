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
from tests.disc.testing_base import DiscTestCase, skipTorchLE

@skipTorchLE("1.6.1")
class TestAtenPower(DiscTestCase):
    def test_tensor_scalar_power(self):
        @torch.jit.script
        def power_func(mat):
            pow_res = mat.size(-1) ** 0.5
            return pow_res

        mat = torch.randn([4, 4], device=self.device)
        if (isinstance(mat, torch.Tensor)):
            mat = (mat.to(self.device),)
        self._test_cvt_to_disc(power_func, mat)

    def test_tensor_tensor_power(self):
        @torch.jit.script
        def power_func(mat):
            exponent = torch.tensor([[0.5, 2.0], [1.3, -2.0]]).to(mat.device)
            pow_res = mat ** exponent
            return pow_res
        mat = torch.tensor([[1, 2], [3, 4]], device=self.device, dtype=torch.int)
        if (isinstance(mat, torch.Tensor)):
            mat = (mat.to(self.device),)
        self._test_cvt_to_disc(power_func, mat)

if __name__ == "__main__":
    unittest.main()

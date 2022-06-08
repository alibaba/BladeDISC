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

import unittest
import torch

from tests.disc.testing_base import DiscTestCase

class TestDiscEngine(DiscTestCase):
    def test_no_output_overwrite(self):
        class Triple(torch.nn.Module):
            def forward(self, x):
                return 3.0 * x + 0.0 + 0.0

        x = torch.randn(1, device=self.device)
        triple = self.cvt_to_disc(Triple().eval(), x)

        one = torch.tensor([1], dtype=torch.float, device=self.device)
        two = torch.tensor([2], dtype=torch.float, device=self.device)

        three = triple(one)
        self.assertEqual(three, 3 * one)

        six = triple(two)
        self.assertEqual(six, 3 * two)

        self.assertEqual(three, 3 * one)


if __name__ == "__main__":
    unittest.main()

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

from typing import List
import unittest

import torch
import torch_blade
from torch_blade import utils

from tests.disc.testing_base import DiscTestCase


class Dummy(torch.nn.Module):
    def forward(self, input, scale, zero_point):
        qmin = -127
        qmax = 128
        num_bits = 8
        axis: List[int] = []
        signed = True
        symatric = True
        dynamic = False
        per_channel = False
        return torch.ops.torch_blade.fake_quant(
                input, scale, zero_point, qmin, qmax, num_bits, axis, signed, symatric, dynamic, per_channel)

@unittest.skipIf(
    utils.torch_version_number() < utils.parse_version("1.8.0") and torch.cuda.is_available(),
    'Quant support since 1.8.0'
)
class TestCustomFakeQuant(DiscTestCase):
    def test_fake_quant(self):
        input = torch.rand(2, 3, 4)
        scale = torch.tensor(0.01)
        zero_point = torch.tensor(0, dtype=torch.int64)
        dummy = Dummy()
        dummy = torch.jit.script(dummy)
        # enlarge atol since it quantization.
        self._test_cvt_to_disc(dummy, (input, scale, zero_point), atol=0.05)


if __name__ == "__main__":
    unittest.main()

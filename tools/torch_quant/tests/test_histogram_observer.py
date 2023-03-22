# Copyright 2022 The BladeDISC Authors. All rights reserved.
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
from torch_quant.observer import HistogramObserver


class TestHistogramObserver(unittest.TestCase):
    def test_basic(self):
        obs = HistogramObserver()
        obs.fake_quant = False
        inp = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)
        oup = obs(inp)
        obs.calculate_qparams()
        self.assertTrue(torch.equal(oup, inp))
        target_scale = torch.tensor(0.01960018463432789)
        target_zp = torch.tensor(0, dtype=torch.int32)
        self.assertTrue(torch.equal(target_scale, obs.scale))
        self.assertTrue(torch.equal(target_zp, obs.zero_point))
        target_fake_quant_output = torch.fake_quantize_per_tensor_affine(
            inp, target_scale, target_zp, 0, 255
        )
        obs.fake_quant = True
        obs.observe = False
        oup = obs(inp)
        self.assertTrue(torch.equal(target_fake_quant_output, oup))

    def test_per_channel(self):
        with self.assertRaises(NotImplementedError):
            HistogramObserver(qscheme=torch.per_channel_symmetric)


if __name__ == "__main__":
    unittest.main()

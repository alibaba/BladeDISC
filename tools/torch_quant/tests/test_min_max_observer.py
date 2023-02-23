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
from torch_quant.observer import MinMaxObserver, PerChannelMinMaxObserver


class MinMaxObserverTest(unittest.TestCase):
    # TODO(litan.ls): more tests for different dtype/qscheme
    def test_basic(self):
        ob = MinMaxObserver(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
        self.assertEqual(ob.qparams.scale, 1)
        self.assertEqual(ob.qparams.zero_point, 0)
        dummy_data = torch.rand((8, 1024)) * 2 - 1
        max_val = torch.max(dummy_data)
        min_val = torch.min(dummy_data)
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
        clip_val = torch.max(-min_val_neg, max_val_pos)
        scale = clip_val / (float(127 + 128) / 2)
        ob(dummy_data)
        self.assertTrue(torch.equal(torch.tensor(scale), ob.qparams.scale))
        self.assertTrue(torch.equal(torch.tensor(0, dtype=torch.int32), ob.qparams.zero_point))


class PerChannelMinMaxObserverTest(unittest.TestCase):
    def test_basic(self):
        ob = PerChannelMinMaxObserver(ch_axis=0, dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
        dummy_data = torch.randn(10, 20)
        ob(dummy_data)
        min_val, max_val = torch.aminmax(dummy_data, dim=1)
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
        clip_val = torch.max(-min_val_neg, max_val_pos)
        scale = clip_val / (float(127 + 128) / 2)
        self.assertTrue(torch.equal(scale, ob.qparams.scale))
        self.assertTrue(torch.equal(torch.zeros_like(scale, dtype=torch.int32), ob.qparams.zero_point))


if __name__ == '__main__':
    unittest.main()

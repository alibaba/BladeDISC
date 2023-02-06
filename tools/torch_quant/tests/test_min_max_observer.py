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
from torch_quant.observer import MinMaxObserver


class MinMaxObserverTest(unittest.TestCase):
    # TODO(litan.ls): more tests for different dtype/qscheme
    def test_basic(self):
        ob = MinMaxObserver(dtype=torch.qint8)
        self.assertEqual(ob.qparams.scale, 1)
        self.assertEqual(ob.qparams.zero_point, 0)
        ob(torch.rand((8, 1024))*2-1)
        torch.testing.assert_close(
            ob.qparams.scale, torch.tensor(1/128), atol=0.01, rtol=0.1)
        torch.testing.assert_close(
            ob.qparams.zero_point, torch.tensor(0.0), atol=1, rtol=0.1)


if __name__ == '__main__':
    unittest.main()

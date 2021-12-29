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
import copy
from torch_blade.testing.common_utils import TestCase
from torch_blade.testing.common_utils import assert_almost_equal

class TestAlmostEqual(TestCase):

    def test_almost_equal(self):
        rect = torch.ones(100, 200)
        square = rect[0:100:2, 0:200:4]

        x = [{"rect": rect, "square": square}, ("x", 2, 3.0), [True, 2, "y"]]
        y = copy.deepcopy(x)
        assert_almost_equal(x, y)
        y[2] = [True, 2, "x"]
        with self.assertRaisesRegex(AssertionError, ".*Error occurred during comparing list elements 2.*"):
            assert_almost_equal(x, y)
        y[2] = [True, 2, "y"]
        y[1] = ("y", 2, 3.0)
        with self.assertRaisesRegex(AssertionError, ".*Error occurred during comparing list elements 1.*"):
            assert_almost_equal(x, y)


if __name__ == '__main__':
    unittest.main()

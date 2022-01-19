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
from torch_blade.testing.common_utils import TestCase

class TestContiguous(TestCase):

    def test_to_contiguous_not_valid(self):
        rect = torch.ones(100, 200)
        self.assertTrue(rect.is_contiguous())

        square = rect[0:100:2, 0:200:4]
        self.assertTrue(not square.is_contiguous())
        self.assertTrue(square.contiguous().is_contiguous())
        # This test was used to notice the tensor.to(contiguous) is not valid.
        self.assertTrue(not square.to(memory_format=torch.contiguous_format)
                                  .is_contiguous())

if __name__ == '__main__':
    unittest.main()

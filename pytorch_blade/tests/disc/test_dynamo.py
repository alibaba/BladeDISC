# Copyright 2023 The BladeDISC Authors. All rights reserved.
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
import logging
import unittest
from tests.disc.testing_base import DiscTestCase, skipTorchLT

def func(a, b):
    return a + b

def func1(**kwargs):
    return func(a=1, **kwargs)

@skipTorchLT("2.0.0")
class TestDynamoCapture(DiscTestCase):
    def test_capture(self):
        import torch._dynamo as dynamo
        import torch_blade.dynamo
        explain_out = dynamo.explain(func1, b=torch.rand([2]))
        self.assertEqual(len(explain_out[2]), 1)

if __name__ == '__main__':
    unittest.main()

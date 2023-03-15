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
from parameterized import parameterized

from tests.models import SimpleModule
from torch_quant.module import ModuleFilter, copy_and_replace, fx_trace


class CopyAndReplaceTest(unittest.TestCase):
    def test_root(self) -> None:
        model = SimpleModule()
        mapping = fx_trace(model)
        copied = copy_and_replace(model, mapping)
        self.assertIs(copied, mapping[''].gm)

    @parameterized.expand([(['conv'],), (['conv', 'sub', 'linear'],)])
    def test_replace(self, include_names) -> None:
        model = SimpleModule()
        dummy_input = torch.randn((1, 2, 5, 5))
        module_filter = ModuleFilter(include_names=include_names)
        mapping = fx_trace(model, module_filter)
        copied = copy_and_replace(model, mapping)
        self.assertEqual(len(mapping), len(include_names))
        for name in include_names:
            self.assertIs(getattr(copied, name), mapping[name].gm)
        self.assertTrue(torch.equal(model(dummy_input), copied(dummy_input)))


if __name__ == '__main__':
    unittest.main()

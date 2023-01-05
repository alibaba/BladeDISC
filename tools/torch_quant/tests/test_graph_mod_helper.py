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

from tests.models import SimpleModule
from torch_quant.graph import _locate_parent


class GraphModHelperTest(unittest.TestCase):
    def test_locate_parent_simple(self) -> None:
        model = SimpleModule()
        parent, name = _locate_parent(model, 'conv')
        self.assertIs(parent, model)
        self.assertEqual(name, 'conv')

    def test_local_parent_nested(self) -> None:
        model = SimpleModule()
        parent, name = _locate_parent(model, 'conv.ob')
        self.assertIs(parent, model.conv)
        self.assertEqual(name, 'ob')

    def test_local_parent_root(self) -> None:
        model = SimpleModule()
        parent, name = _locate_parent(model, '')
        self.assertIs(parent, model)
        self.assertEqual(name, '')


if __name__ == '__main__':
    unittest.main()

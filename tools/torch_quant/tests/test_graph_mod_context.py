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
import torch.nn as nn
from more_itertools import ilen
from parameterized import parameterized
from tests.models import SimpleModule, create_ctx


class DummyModule(nn.Module):
    ...


class GraphModContextTest(unittest.TestCase):
    def test_nodes_by_module_type(self):
        ctx = create_ctx(SimpleModule())
        self.assertEqual(ilen(ctx.nodes_by_module_type([nn.Conv2d])), 2)
        self.assertEqual(ilen(ctx.nodes_by_module_type(
            [nn.AdaptiveMaxPool2d, nn.Linear])), 2)
        self.assertEqual(ilen(ctx.nodes_by_module_type([nn.Softmax])), 0)
        self.assertEqual(ilen(ctx.nodes_by_module_type([])), 0)

    @parameterized.expand([('dummy', ), ('conv.dummy', )])
    def test_add_module(self, full_path: str) -> None:
        ctx = create_ctx(SimpleModule())
        ctx.add_module(full_path, DummyModule())
        m = ctx.root
        for name in full_path.split('.'):
            self.assertTrue(hasattr(m, name))
            m = getattr(m, name)
        self.assertIsInstance(m, DummyModule)

    @parameterized.expand([('dummy', ), ('conv.dummy', )])
    def test_register_buffer(self, full_path: str) -> None:
        ctx = create_ctx(SimpleModule())
        dummy_tensor = torch.rand((1))
        ctx.register_buffer(full_path, dummy_tensor)
        m = ctx.root
        for name in full_path.split('.'):
            self.assertTrue(hasattr(m, name))
            m = getattr(m, name)
        self.assertIsInstance(m, torch.Tensor)
        self.assertTrue(torch.equal(m, dummy_tensor))
        self.assertIsNot(m, dummy_tensor)

    def test_get_or_create_module(self):
        ctx = create_ctx(SimpleModule())
        self.assertFalse(hasattr(ctx.root.conv, 'conv.dummy'))
        m1 = ctx.get_or_create_module('conv.dummy', DummyModule)
        self.assertIs(m1, ctx.root.conv.dummy)
        m2 = ctx.get_or_create_module('conv.dummy', DummyModule)
        self.assertIs(m1, m2)


if __name__ == '__main__':
    unittest.main()

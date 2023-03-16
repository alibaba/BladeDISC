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
from typing import Tuple

import torch
from parameterized import parameterized
from tests.models import (
    SimpleModule,
    SubModule,
    UntraceableSimpleModule,
    UntraceableSubModule,
)
from torch.fx.proxy import TraceError
from torch_quant.module import ModuleFilter, fx_trace


class UntracableModule(torch.nn.Module):
    def forward(self, x):
        return x if torch.sum(x) == 0 else x + 1


class FxTraceTest(unittest.TestCase):
    @parameterized.expand([
        (torch.nn.Conv2d(4, 4, 3), (1, 4, 5, 5)),
        (torch.nn.Linear(2, 4), (1, 2)),
        (SimpleModule(), (1, 2, 5, 5)),
    ])
    def test_simple_module(self, module: torch.nn.Module, input_size: Tuple[int, ...]) -> None:
        mapping = fx_trace(module)
        self.assertIn('', mapping)
        self.assertEqual(len(mapping), 1)
        self.assertIs(mapping[''].m, module)
        inputs = torch.rand(input_size)
        torch.testing.assert_close(mapping[''].gm(inputs), module(inputs))

    @parameterized.expand([
        (UntracableModule(), ),
    ])
    def test_untracable(self, module: torch.nn.Module) -> None:
        self.assertRaises(TraceError, fx_trace, module)

    def test_filter_include_names(self) -> None:
        module = SimpleModule()
        mapping = fx_trace(module, module_filter=ModuleFilter(include_names=['conv']))
        self.assertIn('conv', mapping)
        self.assertEqual(len(mapping), 1)
        self.assertIs(mapping['conv'].m, module.conv)

    def test_filter_include_classes(self) -> None:
        module = SimpleModule()
        module_filter = ModuleFilter(include_classes=[SubModule])
        mapping = fx_trace(module, module_filter=module_filter)
        self.assertIn('sub', mapping)
        self.assertEqual(len(mapping), 1)
        self.assertIs(mapping['sub'].m, module.sub)

    def _check_is_leaf_module(
        self, name: str, module: torch.nn.Module, graph: torch.fx.Graph
    ):
        targets = [nd.target for nd in graph.nodes if nd.op == 'call_module']
        self.assertIn(name, targets)
        for m, _ in module.named_children():
            self.assertNotIn('.'.join([name, m]), targets)

    def test_filter_exclude_names(self) -> None:
        module = SimpleModule()
        module_filter = ModuleFilter(exclude_names=['sub'])
        mapping = fx_trace(module, module_filter=module_filter)
        self.assertIn('', mapping)
        self.assertEqual(len(mapping), 1)
        self._check_is_leaf_module('sub', module.sub, mapping[''].gm.graph)

    def test_filter_exclude_classes(self) -> None:
        module = UntraceableSimpleModule()
        module_filter = ModuleFilter(exclude_classes=[UntraceableSubModule])
        mapping = fx_trace(module, module_filter=module_filter)
        self.assertIn('', mapping)
        self.assertEqual(len(mapping), 1)
        self._check_is_leaf_module(
            'untraceable_sub', module.untraceable_sub, mapping[''].gm.graph
        )

    @parameterized.expand(
        [
            (
                ModuleFilter(
                    include_names=['traceable_sub', 'untraceable_sub.linear_relu'],
                    exclude_names=['traceable_sub.sub'],
                ),
            ),
            (
                ModuleFilter(
                    include_names=['traceable_sub', 'untraceable_sub.linear_relu'],
                    exclude_classes=[SubModule],
                ),
            ),
        ]
    )
    def test_filter_include_names_with_exclude_modules(
        self, module_filter: ModuleFilter
    ) -> None:
        module = UntraceableSimpleModule()
        mapping = fx_trace(module, module_filter=module_filter)
        self.assertIn('traceable_sub', mapping)
        self.assertIn('untraceable_sub.linear_relu', mapping)
        self.assertEqual(len(mapping), 2)
        self._check_is_leaf_module(
            'sub', module.traceable_sub.sub, mapping['traceable_sub'].gm.graph
        )

    @parameterized.expand(
        [
            (
                ModuleFilter(
                    include_classes=[SimpleModule],
                    exclude_names=['traceable_sub.sub'],
                ),
            ),
            (
                ModuleFilter(
                    include_classes=[SimpleModule], exclude_classes=[SubModule]
                ),
            ),
        ]
    )
    def test_filter_include_classes_with_exclude_modules(
        self, module_filter: ModuleFilter
    ) -> None:
        module = UntraceableSimpleModule()
        mapping = fx_trace(module, module_filter=module_filter)
        self.assertIn('traceable_sub', mapping)
        self.assertEqual(len(mapping), 1)
        self._check_is_leaf_module(
            'sub', module.traceable_sub.sub, mapping['traceable_sub'].gm.graph
        )

    def test_custom_tracer(self) -> None:
        class CustomTracer(torch.fx.Tracer):
            def is_leaf_module(self, m, module_qualified_name):
                if isinstance(m, UntraceableSubModule):
                    return True
                return super().is_leaf_module(m, module_qualified_name)

        module = UntraceableSimpleModule()
        custom_tracer = CustomTracer()
        mapping = fx_trace(module, tracer=custom_tracer)
        self.assertIn('', mapping)
        self.assertEqual(len(mapping), 1)
        self._check_is_leaf_module(
            'untraceable_sub', module.untraceable_sub, mapping[''].gm.graph
        )


if __name__ == '__main__':
    unittest.main()

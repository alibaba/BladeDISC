import unittest
from typing import Tuple

import torch
from parameterized import parameterized
from tests.models import SimpleModule
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

    @unittest.skip('not implemented')
    def test_filter_include_name(self) -> None:
        module = SimpleModule()
        mapping = fx_trace(
            module, module_filter=ModuleFilter(include_names=['conv']))
        self.assertIn('conv', mapping)
        self.assertEqual(len(mapping), 1)
        self.assertIs(mapping['conv'], module.conv1)


if __name__ == '__main__':
    unittest.main()

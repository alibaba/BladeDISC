import unittest

from tests.models import SimpleModule
from torch_quant.module import copy_and_replace, fx_trace


class CopyAndReplaceTest(unittest.TestCase):
    def test_root(self) -> None:
        model = SimpleModule()
        mapping = fx_trace(model)
        copied = copy_and_replace(model, mapping)
        self.assertIs(copied, mapping[''].gm)

    @unittest.skip('not implemented')
    def test_replace_single(self) -> None:
        ...

    @unittest.skip('not implemented')
    def test_replace_multiple(self) -> None:
        ...


if __name__ == '__main__':
    unittest.main()

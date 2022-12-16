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

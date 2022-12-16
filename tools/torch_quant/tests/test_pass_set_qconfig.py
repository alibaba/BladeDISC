import unittest

from tests.models import SimpleModule, create_ctx

from torch_quant.graph import QUANTIZABLE_MODULE_TYPES, set_qconfig


class SetQconfigTest(unittest.TestCase):
    def test_basic(self):
        ctx = create_ctx(SimpleModule())
        set_qconfig(ctx)
        for m in ctx.root.modules():
            if any(isinstance(m, t) for t in QUANTIZABLE_MODULE_TYPES):
                self.assertTrue(hasattr(m, 'qconfig'))


if __name__ == '__main__':
    unittest.main()

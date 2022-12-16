import unittest

from tests.models import SimpleModule, create_ctx
from torch_quant.graph import QUANTIZABLE_MODULE_TYPES, insert_act_observer
from torch_quant.observer import Observer


class InsertActObserverTest(unittest.TestCase):
    def test_basic(self) -> None:
        ctx = create_ctx(SimpleModule())
        insert_act_observer(ctx)
        for node in ctx.nodes_by_module_type(QUANTIZABLE_MODULE_TYPES):
            self.assertTrue(any(isinstance(ctx.modules.get(
                arg.target), Observer) for arg in node.args))


if __name__ == '__main__':
    unittest.main()

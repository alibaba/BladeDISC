import torch
import unittest

from torch_addons import mlir

from tests.mlir.testing_utils import DiscTestCase


class TestDiscTools(DiscTestCase):

    def test_const_tensor(self):
        gstr = """
        graph(%x.1 : Tensor,
              %y.1 : Tensor):
          %4 : Tensor = aten::mul(%x.1, %y.1)
          return (%4)"""

        graph = torch._C.parse_ir(gstr)
        all_unspt = all(not mlir.is_mlir_mhlo_supported(n) for n in graph.nodes())
        self.assertTrue(all_unspt)


if __name__ == "__main__":
    unittest.main()

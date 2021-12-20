import torch
import unittest

from typing import List
from tests.mlir.testing_utils import DiscTestCase


class TestDiscBlockOps(DiscTestCase):
    def test_fuse_sub_block(self):
        @torch.jit.script
        def select_fun(tensor) -> List[torch.Tensor]:
            iteration = tensor.numel()
            values = []
            for _ in range(iteration):
                val = tensor + tensor
                values.append(val)
            return values

        x = torch.randn(10).to(self.device)
        test_data = (x,)
        self._test_cvt_to_disc(select_fun, test_data)


if __name__ == "__main__":
    unittest.main()

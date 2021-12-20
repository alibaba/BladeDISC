import unittest
import torch

from tests.mlir.testing_utils import DiscTestCase

class TestDiscEngine(DiscTestCase):
    def test_no_output_overwrite(self):
        class Triple(torch.nn.Module):
            def forward(self, x):
                return 3.0 * x + 0.0 + 0.0

        x = torch.randn(1, device=self.device)
        triple = self.cvt_to_disc(Triple().eval(), x)

        one = torch.tensor([1], dtype=torch.float, device=self.device)
        two = torch.tensor([2], dtype=torch.float, device=self.device)

        three = triple(one)
        self.assertEqual(three, 3 * one)

        six = triple(two)
        self.assertEqual(six, 3 * two)

        self.assertEqual(three, 3 * one)


if __name__ == "__main__":
    unittest.main()

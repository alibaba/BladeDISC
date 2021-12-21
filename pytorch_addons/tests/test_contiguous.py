import torch
import unittest
from torch_addons.testing.common_utils import TestCase

class TestContiguous(TestCase):

    def test_to_contiguous_not_valid(self):
        rect = torch.ones(100, 200)
        self.assertTrue(rect.is_contiguous())

        square = rect[0:100:2, 0:200:4]
        self.assertTrue(not square.is_contiguous())
        self.assertTrue(square.contiguous().is_contiguous())
        # This test was used to notice the tensor.to(contiguous) is not valid.
        self.assertTrue(not square.to(memory_format=torch.contiguous_format)
                                  .is_contiguous())

if __name__ == '__main__':
    unittest.main()

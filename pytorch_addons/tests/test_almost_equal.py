import torch
import unittest
import copy
from torch_addons.testing.common_utils import TestCase
from torch_addons.testing.common_utils import assert_almost_equal

class TestAlmostEqual(TestCase):

    def test_almost_equal(self):
        rect = torch.ones(100, 200)
        square = rect[0:100:2, 0:200:4]

        x = [{"rect": rect, "square": square}, ("x", 2, 3.0), [True, 2, "y"]]
        y = copy.deepcopy(x)
        assert_almost_equal(x, y)
        y[2] = [True, 2, "x"]
        with self.assertRaisesRegex(AssertionError, ".*Error occurred during comparing list elements 2.*"):
            assert_almost_equal(x, y)
        y[2] = [True, 2, "y"]
        y[1] = ("y", 2, 3.0)
        with self.assertRaisesRegex(AssertionError, ".*Error occurred during comparing list elements 1.*"):
            assert_almost_equal(x, y)


if __name__ == '__main__':
    unittest.main()

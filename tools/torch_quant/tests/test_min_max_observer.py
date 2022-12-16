import unittest

import torch
from torch_quant.observer import MinMaxObserver


class MinMaxObserverTest(unittest.TestCase):
    def test_basic(self):
        ob = MinMaxObserver(dtype=torch.qint8)
        self.assertEqual(ob.qparams.scale, 1)
        self.assertEqual(ob.qparams.zero_point, 0)
        ob(torch.rand((8, 1024))*2-1)
        torch.testing.assert_close(
            ob.qparams.scale, torch.tensor(1/128), atol=0.01, rtol=0.1)
        torch.testing.assert_close(
            ob.qparams.zero_point, torch.tensor(0.0), atol=1, rtol=0.1)


if __name__ == '__main__':
    unittest.main()

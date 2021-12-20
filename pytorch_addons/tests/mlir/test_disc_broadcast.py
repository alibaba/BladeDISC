import torch
import unittest

from tests.mlir.testing_utils import DiscTestCase


class TestDiscBroadcast(DiscTestCase):

    def _test_rank0_broadcast(self, broadcast_func):
        test_data = torch.randn([]).to(self.device)
        self._test_cvt_to_disc(broadcast_func, (test_data,))

    def _test_broadcast(self, broadcast_func):
        test_data = torch.randn([3, 1, 1]).to(self.device)
        self._test_cvt_to_disc(broadcast_func, (test_data,))

    def test_expand(self):
        @torch.jit.script
        def expand_func(x):
            return x.expand([2, 3, 224, -1])
        self._test_broadcast(expand_func)

        @torch.jit.script
        def expand_func(x):
            return x.expand([2, 3, 224, 1])
        self._test_broadcast(expand_func)
        self._test_rank0_broadcast(expand_func)

    def test_expand_as(self):
        @torch.jit.script
        def expand_func(x):
            y = torch.ones([2, 3, 22, 22])
            return x.expand_as(y)
        self._test_broadcast(expand_func)
        self._test_rank0_broadcast(expand_func)

    def test_repeat(self):
        @torch.jit.script
        def repeat_func(x):
            return x.repeat(2, 4, 1, 5)
        self._test_broadcast(repeat_func)
        self._test_rank0_broadcast(repeat_func)


if __name__ == '__main__':
    unittest.main()

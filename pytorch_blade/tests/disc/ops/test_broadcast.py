# Copyright 2021 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import unittest

from tests.disc.testing_base import DiscTestCase


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

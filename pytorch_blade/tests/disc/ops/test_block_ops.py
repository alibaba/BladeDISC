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

from typing import List
from tests.disc.testing_base import DiscTestCase


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

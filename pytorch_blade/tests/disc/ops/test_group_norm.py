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

from torch_blade.version import cuda_available
from tests.disc.testing_base import DiscTestCase

class TestDiscGroupNorm(DiscTestCase):
    def _test_group_norm(self, groupnorm):
        test_data = torch.randn([2, 320, 64, 64], device=self.device)
        annotation = ([-1, -1, -1, -1], torch.float)
        self._test_disc(groupnorm, [annotation], (test_data,), rtol=1e-3)

    def test_groupnorm_module(self):
        groupnorm = torch.nn.GroupNorm(32, 320, affine=False)
        self._test_group_norm(groupnorm)

    def test_groupnorm_module_has_affine(self):
        groupnorm = torch.nn.GroupNorm(32, 320, affine=True)
        self._test_group_norm(groupnorm)

if __name__ == "__main__":
    unittest.main()

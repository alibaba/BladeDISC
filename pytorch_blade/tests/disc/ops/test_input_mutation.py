# Copyright 2023 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
from typing import List, Optional, Tuple
from torch import Tensor
import torch_blade
import unittest
from tests.disc.testing_base import skipTorchLE
import torch_blade.clustering.support_fusion_group as fusion
from tests.disc.testing_base import DiscTestCase

class TestInputMutation(DiscTestCase):
    def setUp(self):
        super().setUp()
        os.environ["TORCH_MHLO_OP_WHITE_LIST"] = "aten::copy_;aten::add"

    def tearDown(self):
        del os.environ["TORCH_MHLO_OP_WHITE_LIST"]

    @skipTorchLE("2.0.0")
    def test_input_mutation(self):
        def func(add : Tensor, value : Tensor) -> Tensor:
            add.add_(value)
            return add
        
        with fusion.min_group_nodes(1):
            opt_func = torch.compile(backend='aot_disc')(func)
            add = torch.randn(8, 64, device=self.device)
            value = torch.randn(8, 64, device=self.device)
            actual = opt_func(add.clone(), value.clone())
            expect = func(add.clone(), value.clone())
            self.assertTrue(torch.allclose(actual.cpu(), expect.cpu()))

if __name__ == "__main__":
    unittest.main()

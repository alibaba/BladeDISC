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
import torch.nn as nn
from typing import List, Optional, Tuple
from torch import Tensor
import torch_blade
import unittest
import torch_blade.clustering.support_fusion_group as fusion
from tests.disc.testing_base import skipTorchLE, DiscTestCase

class KVCacheModule(nn.Module):
    def forward(self, k_cache: Tensor, k: Tensor, step : Tensor):
        k_cache[..., step - k.shape[-2]: step , :].add_(k)
        return k_cache

class TestInputMutation(DiscTestCase):
    def setUp(self):
        super().setUp()
        os.environ["TORCH_MHLO_OP_WHITE_LIST"] = "aten::copy_;aten::add;aten::slice_scatter;"
        os.environ["TORCH_BLADE_EXPERIMENTAL_MERGE_HORIZONTAL_GROUPS"] = "true"

    def tearDown(self):
        del os.environ["TORCH_MHLO_OP_WHITE_LIST"]
        del os.environ["TORCH_BLADE_EXPERIMENTAL_MERGE_HORIZONTAL_GROUPS"]

    @skipTorchLE("1.10.0")
    def test_inplace_kv(self):
        k_cache = torch.zeros(2, 32, 8, device=self.device)
        k = torch.ones(2, 1, 8, device=self.device)

        m = KVCacheModule()
        m.train(False)
        step = torch.tensor(1)
        opt_func = torch_blade.optimize(m, allow_tracing=True, model_inputs=(k_cache.clone(), k.clone(), step))
        expect = m(k_cache.clone(), k.clone(), step)
        actual = opt_func(k_cache.clone(), k.clone(), step)
        self.assertTrue(torch.allclose(expect.cpu(), actual.cpu()))


if __name__ == "__main__":
    unittest.main()

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

class TestInputMutation(DiscTestCase):
    def setUp(self):
        super().setUp()
        os.environ["TORCH_MHLO_OP_WHITE_LIST"] = "aten::copy_;aten::add;aten::slice_scatter;"

    def tearDown(self):
        del os.environ["TORCH_MHLO_OP_WHITE_LIST"]
        pass
        
    @skipTorchLE("2.0.0")
    def notest_inplace_kv_cache(self):
        def func(k_cache: Tensor, k: Tensor) -> Tensor:
            k_cache[...,-k.shape[-2] :, :].add_(k)
            return k_cache
        
        with fusion.min_group_nodes(1):
            opt_func = torch.compile(backend='aot_disc')(func)
            add = torch.zeros(2, 8, 32, 2, device=self.device)
            value = torch.ones(2, 8, 1, 2, device=self.device)
            actual = opt_func(add.clone(), value.clone())
            expect = func(add.clone(), value.clone())
            self.assertTrue(torch.allclose(actual.cpu(), expect.cpu()))

    def test_ts_inplace_kv(self):
        import torch.nn as nn
        class TestModule(nn.Module):
            def forward(self, k_cache: Tensor, k: Tensor) -> Tensor:
                # TODO(yancey) supports lowering mhlo.dynamic_update_slice on dynamic shape
                k_cache[-1 : , :].add_(k)
                return k_cache
        from torch_blade.config import Config
        add = torch.zeros(8, 32, device=self.device)
        value = torch.ones(1, 32, device=self.device)
        m = TestModule()
        m.train(False)
        traced = torch.jit.trace(m, (add.clone(), value.clone())).eval()
        opt_func = torch_blade.optimize(m, allow_tracing=True, model_inputs=(add.clone(), value.clone()))
        expect = m(add.clone(), value.clone())
        actual = opt_func(add.clone(), value.clone())
        self.assertTrue(torch.allclose(actual.cpu(), expect.cpu()))


if __name__ == "__main__":
    unittest.main()

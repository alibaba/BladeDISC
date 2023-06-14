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

class TestDiscLayerNorm(DiscTestCase):
    def _test_layer_norm(self, layernorm):
        test_data = torch.randn([2, 3, 224, 224], device=self.device)
        annotation = ([-1, -1, 224, 224], torch.float)
        self._test_disc(layernorm, [annotation], (test_data,))

    def test_layernorm_module(self):
        layernorm = torch.nn.LayerNorm([224, 224], elementwise_affine=False)
        self._test_layer_norm(layernorm)

    def test_layernorm_module_has_affine(self):
        layernorm = torch.nn.LayerNorm([224, 224], elementwise_affine=True)
        self._test_layer_norm(layernorm)

    def test_layernorm_func(self):
        @torch.jit.script
        def layernorm(x):
            reduce_dim = [2, 3]
            mean_x = x.mean(dim=reduce_dim, keepdim=True)
            zero_x = x - mean_x
            square_x = zero_x * zero_x
            var_x = square_x.mean(dim=reduce_dim, keepdim=True)
            var_bias = var_x + 1e-5
            rsqrt_var = var_bias.rsqrt()
            norm_x = zero_x * rsqrt_var
            return norm_x

        @torch.jit.script
        def layernorm_std(x):
            reduce_dim = [2, 3]
            mean_x = x.mean(dim=reduce_dim, keepdim=True)
            zero_x = x - mean_x
            var_x = x.std(dim=reduce_dim, keepdim=True)
            var_bias = var_x + 1e-5
            rsqrt_var = var_bias.rsqrt()
            norm_x = zero_x * rsqrt_var
            return norm_x

        self._test_layer_norm(layernorm)
        self._test_layer_norm(layernorm_std)

if __name__ == "__main__":
    unittest.main()

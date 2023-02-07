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

from torch_blade import utils
from tests.disc.testing_base import DiscTestCase


class TestFactoryLikes(DiscTestCase):

    def _test_factory_like(self, func, d0: int, d1: int, d_or_val: int, rtol: float=1e-6, atol: float=1e-3):
        class TestModel(torch.nn.Module):

            def __init__(self, device):
                super().__init__()
                self.device = device

            def forward(self, d0: int, d1: int, d_or_val: int):
                return func(d0, d1, d_or_val, device=self.device)

        model = torch.jit.script(TestModel(self.device))
        self._test_cvt_to_disc(model, (d0, d1, d_or_val), rtol=rtol, atol=atol)

    def test_factory_likes(self):
        self._test_factory_like(torch.zeros, 2, 2, 3)
        self._test_factory_like(torch.ones, 2, 2, 3)

        if utils.torch_version_number() <= utils.parse_version("1.6.1"):
            return

        @torch.jit.script
        def fulls(d0: int, d1: int, val: int, device: torch.device):
            return torch.full([d0, d1], val, device=device)

        @torch.jit.script
        def fulls_dtype(d0: int, d1: int, val: int, device: torch.device):
            return torch.full([d0, d1], val,
                              dtype=torch.float32, device=device)

        self._test_factory_like(fulls, 2, 2, 3)
        self._test_factory_like(fulls_dtype, 2, 2, 3)
        
        if utils.torch_version_number() <= utils.parse_version("1.8.1"):
            return

        @torch.jit.script
        def empty(d0: int, d1: int, val: int, device: torch.device):
            return torch.empty([d0, d1], device=device)

        @torch.jit.script
        def empty_dtype(d0: int, d1: int, val: int, device: torch.device):
            return torch.empty([d0, d1],
                              dtype=torch.float32, device=device)

        # results are not numeric checkable
        self._test_factory_like(empty, 2, 2, 3, rtol=1e+6, atol=1e+6)
        self._test_factory_like(empty_dtype, 2, 2, 3, rtol=1e+6, atol=1e+6)

if __name__ == "__main__":
    unittest.main()

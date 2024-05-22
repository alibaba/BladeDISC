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
from tests.disc.testing_base import DiscTestCase, skipTorchLE

@skipTorchLE("1.6.1")
class TestAtenScatter(DiscTestCase):
    
    def test_scatter(self):
        if self.device != torch.device('cuda'):
            return
        
        @torch.jit.script
        def scatter_func(destination, place_at, source):
            a = torch.scatter(destination, 0, place_at, source)
            return a

        destination = torch.rand(3, 5, dtype=torch.float32, device=self.device)

        source = torch.rand(4, 5, dtype=torch.float32, device=self.device)
        indices = torch.randint(0, 3, (2, 4), dtype=torch.int64, device=self.device)

        annotations = [(list(destination.shape), torch.float32), (list(
            indices.shape), torch.int64), (list(source.shape), torch.float32)]
        self._test_disc(scatter_func, annotations,
                        (destination, indices, source))

    def test_scatteradd(self):
        if self.device != torch.device('cuda'):
            return

        @torch.jit.script
        def scatter_func(destination, place_at, source):
            a = torch.scatter_add(destination, 0, place_at, source)
            return a

        destination = torch.rand(3, 5, dtype=torch.float32, device=self.device)
        source = torch.rand(2, 5, dtype=torch.float32, device=self.device)
        indices = torch.randint(0, 3, (2, 4), dtype=torch.int64, device=self.device)

        annotations = [(list(destination.shape), torch.float32), (list(
            indices.shape), torch.int64), (list(source.shape), torch.float32)]
        self._test_disc(scatter_func, annotations,
                        (destination, indices, source))


if __name__ == "__main__":
    unittest.main()

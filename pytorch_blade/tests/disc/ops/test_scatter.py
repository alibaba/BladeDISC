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

        @torch.jit.script
        def scatter_func(destination, place_at, source):
            a = torch.scatter(destination, 0, place_at, source)
            return a

        destination = torch.tensor(
            [
                [4.0, 0.0, 3.0, 1.0, 0.0],
                [0.0, 5.0, 8.0, 0.0, 0.0],
                [6.0, 0.0, 0.0, 9.0, 0.0]
            ], dtype=torch.float32, device=self.device)

        source = torch.tensor(
            [
                [0.3992,  0.2908,  0.9044,  0.4850,  0.6004],
                [0.5735,  0.9006,  0.6797,  0.4152,  0.1732]
            ], dtype=torch.float32, device=self.device)

        place_at = torch.tensor(
            [
                [0, 1, 2, 0],
                [2, 0, 0, 1]
            ], dtype=torch.int64, device=self.device)

        annotations = [(list(destination.shape), torch.float32), (list(
            place_at.shape), torch.int64), (list(source.shape), torch.float32)]
        self._test_disc(scatter_func, annotations,
                        (destination, place_at, source))
    
    def test_scatteradd(self):

        @torch.jit.script
        def scatter_func(destination, place_at, source):
            a = torch.scatter_add(destination, 0, place_at, source)
            return a

        destination = torch.tensor(
            [
                [4.0, 0.0, 3.0, 1.0, 0.0],
                [0.0, 5.0, 8.0, 0.0, 0.0],
                [6.0, 0.0, 0.0, 9.0, 0.0]
            ], dtype=torch.float32, device=self.device)

        source = torch.tensor(
            [
                [0.3992,  0.2908,  0.9044,  0.4850,  0.6004],
                [0.5735,  0.9006,  0.6797,  0.4152,  0.1732]
            ], dtype=torch.float32, device=self.device)

        place_at = torch.tensor(
            [
                [0, 1, 2, 0],
                [2, 0, 0, 1]
            ], dtype=torch.int64, device=self.device)

        annotations = [(list(destination.shape), torch.float32), (list(
            place_at.shape), torch.int64), (list(source.shape), torch.float32)]
        self._test_disc(scatter_func, annotations,
                        (destination, place_at, source))


if __name__ == "__main__":
    unittest.main()

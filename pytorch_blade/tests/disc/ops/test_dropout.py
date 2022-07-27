# Copyright 2022 The BladeDISC Authors. All rights reserved.
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
import torch_blade
import unittest

from tests.disc.testing_base import DiscTestCase

class TestMlirConvolution(DiscTestCase):
    def test_native_dropout(self):
        @torch.jit.script
        def jit_script_func(x):
            return torch.native_dropout(x, 1.0, True)
        self._test_disc(jit_script_func, [([-1, -1, -1], torch.float)])
        self.assertTrue(False)

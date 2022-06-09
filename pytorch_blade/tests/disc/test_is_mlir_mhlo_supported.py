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

from torch_blade import mlir

from tests.disc.testing_base import DiscTestCase


class TestDiscTools(DiscTestCase):

    def test_const_tensor(self):
        gstr = """
        graph(%x.1 : Tensor,
              %y.1 : Tensor):
          %4 : Tensor = aten::mul(%x.1, %y.1)
          return (%4)"""

        graph = torch._C.parse_ir(gstr)
        all_unspt = all(not mlir.is_mlir_mhlo_supported(n) for n in graph.nodes())
        self.assertTrue(all_unspt)


if __name__ == "__main__":
    unittest.main()

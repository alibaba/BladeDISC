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

import torch
import torch_blade
import unittest
from torch.testing import FileCheck

class TestDiscAttentionOp(unittest.TestCase):
    def test_attentionF_graph(self):
        @torch.jit.script
        def func_attF(query, key, value, attn_mask):
            return torch.ops.disc.attentionF(query, key, value, attn_mask, "NHWC", "NHWC", False, False)
        expected_gstr = """
graph(%query.1 : Tensor,
      %key.1 : Tensor,
      %value.1 : Tensor):
  # CHECK: disc::attentionF
  %8 : Tensor, %9 : Tensor = disc::attentionF(%query.1, %key.1, %value.1, %6, %7)
  return (%10)
"""
        FileCheck().run(expected_gstr, str(func_attF.graph))

if __name__ == "__main__":
    unittest.main()
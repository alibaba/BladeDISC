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

import unittest

from tests.models import SimpleModule, create_ctx
from torch_quant.graph import QUANTIZABLE_MODULE_TYPES, set_qconfig


class SetQconfigTest(unittest.TestCase):
    def test_basic(self):
        ctx = create_ctx(SimpleModule())
        set_qconfig(ctx)
        for m in ctx.root.modules():
            if any(isinstance(m, t) for t in QUANTIZABLE_MODULE_TYPES):
                self.assertTrue(hasattr(m, 'qconfig'))


if __name__ == '__main__':
    unittest.main()

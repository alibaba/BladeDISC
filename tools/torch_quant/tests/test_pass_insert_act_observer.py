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
from torch_quant.graph import QUANTIZABLE_MODULE_TYPES, insert_act_observer
from torch_quant.observer import Observer


class InsertActObserverTest(unittest.TestCase):
    def test_basic(self) -> None:
        ctx = create_ctx(SimpleModule())
        insert_act_observer(ctx)
        for node in ctx.nodes_by_module_type(QUANTIZABLE_MODULE_TYPES):
            self.assertTrue(any(isinstance(ctx.modules.get(
                arg.target), Observer) for arg in node.args))


if __name__ == '__main__':
    unittest.main()

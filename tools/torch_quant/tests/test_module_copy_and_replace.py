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

from tests.models import SimpleModule
from torch_quant.module import copy_and_replace, fx_trace


class CopyAndReplaceTest(unittest.TestCase):
    def test_root(self) -> None:
        model = SimpleModule()
        mapping = fx_trace(model)
        copied = copy_and_replace(model, mapping)
        self.assertIs(copied, mapping[''].gm)

    @unittest.skip('not implemented')
    def test_replace_single(self) -> None:
        ...

    @unittest.skip('not implemented')
    def test_replace_multiple(self) -> None:
        ...


if __name__ == '__main__':
    unittest.main()

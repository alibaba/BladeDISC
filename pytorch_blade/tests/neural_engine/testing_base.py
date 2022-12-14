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

from torch_blade.neural_engine import is_available
from torch_blade.testing.common_utils import TestCase


class NeuralEngineTestCase(TestCase):
    def setUp(self):
        super().setUp()
        self.is_neural_engine_available = is_available()
        if not self.is_neural_engine_available:
            self.skipTest("Neural engine is not built")

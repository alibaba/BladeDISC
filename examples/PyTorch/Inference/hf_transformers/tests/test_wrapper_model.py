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

import unittest

from blade_adapter import WrapperModel, create_model, trace_model
from transformers import GPT2LMHeadModel


class WrapperModelTest(unittest.TestCase):
    def test_create_wrapper_model(self):
        model, info = create_model(id_or_path='gpt2', torchscript=True)
        traced = trace_model(model, info)
        wrapper_model = WrapperModel.create_wrapper_model(traced, info)
        self.assertEqual(type(wrapper_model).__name__,
                         'GPT2LMHeadModelBladeWrapper')
        self.assertIsInstance(wrapper_model, GPT2LMHeadModel)


if __name__ == '__main__':
    unittest.main()

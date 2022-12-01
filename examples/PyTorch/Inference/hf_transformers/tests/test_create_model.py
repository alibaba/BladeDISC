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

from blade_adapter import create_model


class CreateModelTest(unittest.TestCase):
    def test_no_task_no_model(self) -> None:
        self.assertRaises(ValueError, create_model)

    def test_task(self) -> None:
        model, info = create_model(task='feature-extraction')
        self.assertIsInstance(model, info.model_class)
        self.assertEqual(model.config, info.config)
        self.assertEqual(info.task, 'feature-extraction')
        self.assertEqual(info.id_or_path, 'distilbert-base-cased')

    def test_id_or_path(self) -> None:
        model, info = create_model(id_or_path='gpt2')
        self.assertIsInstance(model, info.model_class)
        self.assertEqual(model.config, info.config)
        self.assertEqual(info.task, 'text-generation')
        self.assertEqual(info.id_or_path, 'gpt2')


if __name__ == '__main__':
    unittest.main()

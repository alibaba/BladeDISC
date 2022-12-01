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

from blade_adapter import pipeline
from transformers import DistilBertForSequenceClassification


class PipelineTest(unittest.TestCase):
    def test_task_default(self) -> None:
        pipe = pipeline(task="text-classification")
        output = pipe("This restaurant is awesome")
        self.assertTrue(output)
        self.assertEqual(output[0]['label'], 'POSITIVE')
        self.assertGreater(output[0]['score'], 0.9)

    def test_task_model_id(self) -> None:
        pipe = pipeline(
            model="distilbert-base-uncased-finetuned-sst-2-english")
        output = pipe("This restaurant is awesome")
        self.assertTrue(output)
        self.assertEqual(output[0]['label'], 'POSITIVE')
        self.assertGreater(output[0]['score'], 0.9)

    def test_preloaded_model(self) -> None:
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased-finetuned-sst-2-english', torchscript=True)
        pipe = pipeline(task="text-classification", model=model)
        output = pipe("This restaurant is awesome")
        self.assertTrue(output)
        self.assertEqual(output[0]['label'], 'POSITIVE')
        self.assertGreater(output[0]['score'], 0.9)


if __name__ == '__main__':
    unittest.main()

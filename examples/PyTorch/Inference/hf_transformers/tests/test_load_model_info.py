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

import torch
from blade_adapter import load_model_info
from transformers import (AutoConfig, BartForConditionalGeneration,
                          BertForMaskedLM, DistilBertForSequenceClassification,
                          PretrainedConfig)


class LoadModelInfoTest(unittest.TestCase):
    def test_no_task_no_model(self) -> None:
        self.assertRaises(ValueError, load_model_info)

    def test_task(self) -> None:
        info = load_model_info(task='text-classification')
        self.assertEqual(info.task, 'text-classification')
        self.assertEqual(
            info.id_or_path, 'distilbert-base-uncased-finetuned-sst-2-english')
        self.assertIsInstance(info.config, PretrainedConfig)
        self.assertEqual(info.config._name_or_path, info.id_or_path)
        self.assertIs(info.model_class, DistilBertForSequenceClassification)

    def test_id_or_path(self) -> None:
        info = load_model_info(id_or_path='bert-base-uncased')
        self.assertEqual(info.task, 'fill-mask')
        self.assertEqual(info.id_or_path, 'bert-base-uncased')
        self.assertIsInstance(info.config, PretrainedConfig)
        self.assertEqual(info.config._name_or_path, info.id_or_path)
        self.assertIs(info.model_class, BertForMaskedLM)

    def test_config(self) -> None:
        config = AutoConfig.from_pretrained('facebook/bart-large-cnn')
        info = load_model_info(task='summarization', config=config)
        self.assertEqual(info.task, 'summarization')
        self.assertEqual(info.id_or_path, 'facebook/bart-large-cnn')
        self.assertIs(info.config, config)
        self.assertEqual(info.config._name_or_path, info.id_or_path)
        self.assertIs(info.model_class, BartForConditionalGeneration)

    def test_example_inputs(self) -> None:
        self.assertRaises(ValueError, load_model_info, task='text-generation', example_inputs={
            'wrong_input_key': torch.Tensor([0])
        })
        default_info = load_model_info(task='text-generation')
        info = load_model_info(task='text-generation', example_inputs={
            'input_ids': torch.tensor([[1234]], dtype=torch.int64),
        })
        self.assertNotEqual(
            info.example_inputs['input_ids'].shape, default_info.example_inputs['input_ids'].shape)

    def test_output_names(self) -> None:
        default_info = load_model_info(task='text-generation')
        self.assertEqual(default_info.output_names, ['logits'])
        info = load_model_info(task='text-generation', output_names=[
            'logits', 'past_key_values'
        ])
        self.assertNotEqual(default_info.output_names, info.output_names)


if __name__ == '__main__':
    unittest.main()

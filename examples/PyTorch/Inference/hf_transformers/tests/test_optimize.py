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

import os
import unittest

import torch
from blade_adapter import WrapperModel, optimize
from transformers import AutoModelForMaskedLM, PreTrainedModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"


class OptimizeTest(unittest.TestCase):
    def test_no_task_no_model(self):
        self.assertRaises(ValueError, optimize)

    def _eval_models(self, blade_model: WrapperModel, eager_model: PreTrainedModel) -> None:
        inputs = blade_model.info.example_inputs
        blade_outputs = blade_model(**inputs)
        eager_outputs = eager_model(**inputs)
        self.assertTrue(blade_outputs)
        for x in blade_outputs:
            torch.testing.assert_close(
                blade_outputs[x].float(), eager_outputs[x].float(), rtol=0.001, atol=0.001)

    def test_task_default(self):
        blade_model = optimize(task='fill-mask')
        eager_model = AutoModelForMaskedLM.from_pretrained(
            blade_model.info.id_or_path)
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        eager_model.to(device).eval()
        self._eval_models(blade_model, eager_model)

    def test_id_or_path(self):
        blade_model = optimize(id_or_path='bert-base-uncased')
        eager_model = AutoModelForMaskedLM.from_pretrained(
            blade_model.info.id_or_path)
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        eager_model.to(device).eval()
        self._eval_models(blade_model, eager_model)

    def test_no_task_with_model(self):
        eager_model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
        self.assertRaises(ValueError, optimize, model=eager_model)

    def test_non_torchcript_model(self):
        eager_model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
        self.assertRaises(ValueError, optimize,
                          task='fill-mask', model=eager_model)

    def test_task_and_model(self):
        eager_model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
        traceable_model = AutoModelForMaskedLM.from_pretrained(
            'bert-base-uncased', torchscript=True)
        blade_model = optimize(task='fill-mask', model=traceable_model)
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        eager_model.to(device).eval()
        self._eval_models(blade_model, eager_model)


if __name__ == '__main__':
    unittest.main()

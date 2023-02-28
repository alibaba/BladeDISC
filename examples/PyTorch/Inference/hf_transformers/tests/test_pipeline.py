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
from typing import Any, Dict, List

import numpy as np
import torch
from blade_adapter import _default_device, pipeline
from parameterized import parameterized
from transformers import DistilBertForSequenceClassification
from transformers import pipeline as hf_pipeline
from transformers import set_seed


class PipelineTest(unittest.TestCase):

    @unittest.skip('debug')
    def test_task_default(self) -> None:
        pipe = pipeline(task="text-classification")
        output = pipe("This restaurant is awesome")
        self.assertTrue(output)
        self.assertEqual(output[0]['label'], 'POSITIVE')
        self.assertGreater(output[0]['score'], 0.9)

    @unittest.skip('debug')
    def test_task_model_id(self) -> None:
        pipe = pipeline(
            model="distilbert-base-uncased-finetuned-sst-2-english")
        output = pipe("This restaurant is awesome")
        self.assertTrue(output)
        self.assertEqual(output[0]['label'], 'POSITIVE')
        self.assertGreater(output[0]['score'], 0.9)

    @unittest.skip('debug')
    def test_preloaded_model(self) -> None:
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased-finetuned-sst-2-english', torchscript=True)
        pipe = pipeline(task="text-classification", model=model)
        output = pipe("This restaurant is awesome")
        self.assertTrue(output)
        self.assertEqual(output[0]['label'], 'POSITIVE')
        self.assertGreater(output[0]['score'], 0.9)

    def assert_pipeline_outputs_equal(self, a, b) -> None:
        self.assertEqual(type(a), type(b))
        if np.asarray(a).dtype.kind == 'f':
            np.testing.assert_allclose(a, b, rtol=0.01, atol=0.01)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for x in a:
                self.assert_pipeline_outputs_equal(a[x], b[x])
        elif isinstance(a, list):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_pipeline_outputs_equal(x, y)
        else:
            self.assertEqual(a, b)

    @parameterized.expand([
        ("feature-extraction", ["hello world."], {}),
        ("text-classification", ["This restaurant is awesome"], {}),
        ("token-classification", ["What's the weather in Shanghai?"], {}),
        ("question-answering", [], {
            "context": "Alice is the CEO. Bob is sitting next to her.",
            "question": "Who is sitting next to CEO?",
        }),
        ("fill-mask", ["The man worked as a <mask>."], {}),
    ])
    def test_basic_pipelines(self, task: str, input_args: List[Any],
                             input_kwargs: Dict[str, Any]) -> None:
        # skip compilation for fast functionality test
        pipe = pipeline(task=task, skip_compile=True)
        output = pipe(*input_args, **input_kwargs)
        hf_pipe = hf_pipeline(task=task, device=_default_device())
        hf_output = hf_pipe(*input_args, **input_kwargs)

        self.assert_pipeline_outputs_equal(output, hf_output)

    @parameterized.expand([
        ({'task': 'text-generation', 'forward_default_kwargs': {'use_cache': False}},
         ["I can't believe you did such a "], {'use_cache': False}),
        ({
            'task': 'text-generation',
            'example_inputs': {
                'input_ids': torch.tensor([[123, 456]], dtype=torch.int64),
                # kv cache as tuple: ((k, v),) * num_layers, k/v shape: [batch, head, seq_len, head_dim]
                'past_key_values': lambda: ((torch.empty((1, 12, 0, 768//12), dtype=torch.float), ) * 2, ) * 12,
                'attention_mask': torch.tensor([[1, 1]], dtype=torch.int32),
            },
            'output_names': ['logits', 'past_key_values'],
            'forward_default_kwargs': {'use_cache': True}
        }, ["I can't believe you did such a "], {'use_cache': True})
        # TODO(litan.ls): support tasks with encoder-decoder model
        # ("summarization", [r'''The tower is 324 metres (1,063 ft) tall,
        #   about the same height as an 81-storey building,
        #   and the tallest structure in Paris.'''], {}),
        # ("translation_en_to_fr", ["Hello!"], {}),
    ])
    def test_seq2seq_pipelines(self, pipeline_kwargs: Dict[str, Any],
                               input_args: List[Any], input_kwargs: Dict[str, Any]) -> None:
        # disable torch jit to avoid dynamic shape overhead
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(False)
        # skip blade compilation for fast functionality test
        pipe = pipeline(skip_compile=True, **pipeline_kwargs)
        set_seed(0)
        output = pipe(*input_args, **input_kwargs)
        hf_pipe = hf_pipeline(
            task=pipeline_kwargs['task'], device=_default_device())
        set_seed(0)
        hf_output = hf_pipe(*input_args, **input_kwargs)

        self.assert_pipeline_outputs_equal(output, hf_output)


if __name__ == '__main__':
    unittest.main()

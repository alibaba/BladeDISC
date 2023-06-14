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

import torch
from blade_adapter import _kwargs_to_args, create_model, trace_model
from parameterized import parameterized


class TraceModelTest(unittest.TestCase):
    @parameterized.expand([
        'bert-base-uncased',
        'gpt2',
    ])
    def test_trace_converage(self, id_or_path: str) -> None:
        model, info = create_model(id_or_path=id_or_path, torchscript=True)
        traced = trace_model(model, info)
        golden_outputs = model(**info.example_inputs)
        inputs = _kwargs_to_args(info.input_order, **info.example_inputs)
        traced_outputs = traced(*inputs)
        self.assertEqual(len(golden_outputs), len(traced_outputs))
        for a, b in zip(golden_outputs, traced_outputs):
            torch.testing.assert_close(a, b, rtol=0.01, atol=0.01)


if __name__ == '__main__':
    unittest.main()

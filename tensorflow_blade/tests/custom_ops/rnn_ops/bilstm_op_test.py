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
from typing import Dict, List, Tuple

import numpy as np

from tests.custom_ops.tf_blade_ops_ut_common import TfCustomOpsTestCase  # noqa: E402


class BilstmTest(TfCustomOpsTestCase):
    def _test(
        self, feed_data: Dict[str, np.ndarray], expected_output: List[np.ndarray]
    ) -> None:
        output = self.blade_ops.blade_bilstm(  # type: ignore
            input=feed_data["input"],
            input_h=feed_data["input_h"],
            input_c=feed_data["input_c"],
            weight=feed_data["weight"],
        )
        self.assertAllClose(output[0], expected_output[0])
        self.assertAllClose(output[1], expected_output[1])

    def _get_data(self) -> Tuple[Dict[str, np.ndarray], List[np.ndarray]]:
        batch_size = 4
        hidden_num = 8
        input_dim = 6
        num_steps = 3
        feed_data = dict()
        inputs = np.ones((num_steps, batch_size, input_dim), dtype=np.float32)
        input_h = np.zeros((2, batch_size, hidden_num), dtype=np.float32)
        input_c = np.zeros((2, batch_size, hidden_num), dtype=np.float32)
        params_size = 2 * 4 * hidden_num * (input_dim + hidden_num + 2)
        weight = np.ones((params_size), dtype=np.float32)
        feed_data["input"] = inputs
        feed_data["input_h"] = input_h
        feed_data["input_c"] = input_c
        feed_data["weight"] = weight
        output = np.expand_dims(
            [[0.761198, 0.995051], [0.964003, 0.964003], [0.995051, 0.761198]], axis=1
        )
        output = output.repeat([hidden_num, hidden_num], axis=2)
        output = output.repeat([batch_size], axis=1)
        output_h = np.ones((2, batch_size, hidden_num), dtype=np.float32) * 0.995051
        output_c = np.ones((2, batch_size, hidden_num), dtype=np.float32) * 2.999663
        expected_output = [output, output_h, output_c]
        return feed_data, expected_output

    def testBilstm(self) -> None:
        (feed_data, expected_output) = self._get_data()
        self._test(feed_data, expected_output)


if __name__ == "__main__":
    unittest.main()

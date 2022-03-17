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


class FusedLSTMElementWiseTest(TfCustomOpsTestCase):
    def _test(
        self, feed_data: Dict[str, np.ndarray], expected_output: List[np.ndarray]
    ) -> None:
        output = self.blade_ops.blade_fused_lstm_element_wise(  # type: ignore
            inputs=feed_data["inputs"],
            c_in=feed_data["c_in"],
            b_in=feed_data["b_in"],
            forget_bias=feed_data["forget_bias"],
        )
        with self.get_test_session():
            self.assertAllClose(output[0], expected_output[0], atol=1e-3)
            self.assertAllClose(output[1], expected_output[1], atol=1e-3)

    def _get_data(
        self, data_type: np.dtype
    ) -> Tuple[Dict[str, np.ndarray], List[np.ndarray]]:
        batch_size = 4
        hidden_num = 8
        feed_data = dict()
        feed_data["inputs"] = np.ones((batch_size, hidden_num * 4), dtype=data_type)
        feed_data["c_in"] = np.ones((batch_size, hidden_num), dtype=data_type)
        feed_data["b_in"] = np.zeros((hidden_num * 4), dtype=data_type)
        feed_data["forget_bias"] = 0.0
        c_out = np.ones((batch_size, hidden_num), dtype=data_type) * 1.28782851
        h_out = np.ones((batch_size, hidden_num), dtype=data_type) * 0.62765528
        expected_output = [c_out, h_out]
        return feed_data, expected_output

    def testFusedLSTMElementWise(self) -> None:
        for data_type in [np.float16, np.float32]:
            feed_data, expected_output = self._get_data(data_type)
            self._test(feed_data, expected_output)


if __name__ == "__main__":
    unittest.main()

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
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf

from tests.custom_ops.tf_blade_ops_ut_common import TfCustomOpsTestCase  # noqa: E402


class NonMaxSuppressionTest(TfCustomOpsTestCase):
    def _test(
        self, feed_data: Dict[str, np.ndarray], expected_output: List[np.ndarray]
    ) -> None:
        # NOTE(lanbo.llb): We use a fast sort impl on gpus, we test
        # on some cases, this will not affect the accuracy of
        # your model, but the values may not be exactly the same as tf.image.non_max_suppression
        output = self.blade_ops.blade_non_max_suppression(  # type: ignore
            feed_data["boxes:0"], feed_data["scores:0"], max_output_size=8,
        )
        with self.get_test_session():
            self.assertAllClose(output, expected_output[0], atol=1e-3)

    def _get_data(self,) -> Tuple[Dict[str, np.ndarray], List[np.ndarray]]:
        # build the test graph
        tf.compat.v1.reset_default_graph()
        num_box = 2048
        boxes = tf.compat.v1.placeholder(tf.float32, shape=[num_box, 4], name='boxes')
        scores = tf.compat.v1.placeholder(tf.float32, shape=[num_box], name='scores')

        _ = tf.image.non_max_suppression(boxes, scores, max_output_size=8, name='nms',)
        graph_def = tf.compat.v1.get_default_graph().as_graph_def()
        for node in graph_def.node:
            print(node)

        feed_data: Dict[str, Any] = dict()
        feed_data["boxes:0"] = np.random.rand(num_box, 4)
        feed_data["scores:0"] = np.random.rand(num_box)
        with self.get_test_session(graph=tf.compat.v1.get_default_graph()) as sess:
            expected_output = sess.run(["nms/NonMaxSuppressionV3:0"], feed_data)
        return feed_data, expected_output

    def testNonMaxSuppression(self) -> None:
        feed_data, expected_output = self._get_data()
        self._test(feed_data, expected_output)


if __name__ == "__main__":
    unittest.main()

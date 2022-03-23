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
from typing import Any, Dict

import numpy as np

from tf_blade.gpu.tf_non_max_suppression_opt import TfNonMaxSuppressionOpt
from tf_blade.util.tf_import_helper import tf

tf.disable_eager_execution()


class TfNonMaxSuppressionOptTest(unittest.TestCase):
    def test_basic(self) -> None:
        # build the test graph
        tf.reset_default_graph()
        num_box = 2048
        boxes = tf.placeholder(tf.float32, shape=[num_box, 4], name='boxes')
        scores = tf.placeholder(tf.float32, shape=[num_box], name='scores')
        max_output_size = tf.constant(128, dtype=tf.int32, name='max_output_size')

        selected_indices = tf.image.non_max_suppression(
            boxes, scores, max_output_size, iou_threshold=0.5, name='nms'
        )
        _ = tf.gather(boxes, selected_indices)

        fd: Dict[str, Any] = dict()
        fd["boxes:0"] = np.random.rand(num_box, 4)
        fd["scores:0"] = np.random.rand(num_box)

        opt_pass = TfNonMaxSuppressionOpt()
        count, ret_graph_def = opt_pass.optimize_graph_def(
            tf.get_default_graph().as_graph_def(),
        )
        self.assertTrue(count > 0, "pattern not detected")


if __name__ == '__main__':
    unittest.main()

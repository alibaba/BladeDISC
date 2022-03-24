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

from tf_blade.gpu.tf_lstm_element_wise_ops import TfLstmElementWiseOps
from tf_blade.util import tf_graph_transform_util as graph_transform
from tf_blade.util.tf_import_helper import tf

tf.disable_eager_execution()


class TfLstmElementWiseOpsTest(unittest.TestCase):
    INPUT_DIM = 6
    NUM_STEPS = 10
    HIDDEN_NUM = 8

    def build_model(self) -> List[str]:
        lstm_cell = "BasicLSTMCell"
        X = tf.compat.v1.placeholder(
            tf.float32,
            shape=[
                TfLstmElementWiseOpsTest.NUM_STEPS,
                None,
                TfLstmElementWiseOpsTest.INPUT_DIM,
            ],
            name="X",
        )
        X_reshape = tf.reshape(X, [-1, TfLstmElementWiseOpsTest.INPUT_DIM])
        X_list = tf.split(
            axis=0,
            num_or_size_splits=int(TfLstmElementWiseOpsTest.NUM_STEPS),
            value=X_reshape,
        )
        L = tf.compat.v1.placeholder(tf.int32, shape=[None], name="L")
        cell_func = getattr(tf.compat.v1.nn.rnn_cell, lstm_cell)
        lstm_cell = cell_func(
            num_units=TfLstmElementWiseOpsTest.HIDDEN_NUM,
            forget_bias=0.0,
            state_is_tuple=True,
        )
        Y_list, _ = tf.compat.v1.nn.static_rnn(
            cell=lstm_cell, inputs=X_list, sequence_length=L, dtype=tf.float32
        )
        _ = tf.reshape(
            Y_list,
            [
                TfLstmElementWiseOpsTest.NUM_STEPS,
                -1,
                TfLstmElementWiseOpsTest.HIDDEN_NUM,
            ],
            name="output",
        )

        return ["output"]

    def fill_feed_dict(self, feed_dict: Dict[str, Any], batch_size: int = 2) -> None:
        feed_dict["X:0"] = np.ones(
            (
                TfLstmElementWiseOpsTest.NUM_STEPS,
                batch_size,
                TfLstmElementWiseOpsTest.INPUT_DIM,
            ),
            dtype=np.float32,
        )
        feed_dict["L:0"] = np.ones((batch_size), dtype=np.int32) * 3

    def test_basic_lstm_cell(self) -> None:
        # build the test graph
        outputs = self.build_model()
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            graph_def = tf.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, outputs
            )

        feed_dict: Dict[str, Any] = dict()
        self.fill_feed_dict(feed_dict)
        opt_pass = TfLstmElementWiseOps()
        ret_graph_def = opt_pass.optimize_graph_def(graph_def, ["X", "L"], outputs,)
        opt_graph_def_ops = [n.op for n in ret_graph_def.node]
        self.assertTrue(
            graph_transform.OpType.BLADE_FUSED_LSTM_ELEMENT_WISE.value
            in opt_graph_def_ops
        )


if __name__ == "__main__":
    unittest.main()

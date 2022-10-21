#!/usr/bin/env python
# Copyright 2021 The BladeDISC Authors. All rights reserved.
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
import numpy as np
from test.gpu.mlir.base import MlirTestCase
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf
from google.protobuf import text_format

tf.debugging.set_log_device_placement(True)

class TestTranspose(MlirTestCase):
    def setUp(self):
        os.environ["DISC_GPU_ENABLE_TRANSPOSE_LIBRARY_CALL"] = "true"
        MlirTestCase.setUpWithMLIR(self._testMethodName)

    def _check_launch_op(self, fn, launch_op_name):
        with open(fn) as f:
            graph_def = tf.GraphDef()
            text_format.Merge(f.read(), graph_def)

            disc_op = self.get_node(graph_def, launch_op_name)
            mlir_func_name = disc_op.attr["mlir_function"].func.name
            self.assertNotEqual(mlir_func_name, "")

            mlir_func = self.get_func(graph_def, mlir_func_name)
            self.get_node(mlir_func, "Transpose")

    def test_transpose(self):
        input = np.random.rand(6, 7).astype(np.float32)
        g = tf.Graph()
        with g.as_default():
            t1 = tf.placeholder(shape=(6, 7), dtype=np.float32, name="px")
            t2 = tf.transpose(t1)
            with super(TestTranspose, self).new_sess() as sess:
                sess.run(t2, {t1: input})
            self.assertTrue(t2.shape, (7, 6))

        fn = super(TestTranspose, self).dumped_file('after_tao_pass.pbtxt')
        if self.is_platform_alibaba():
            self._check_launch_op(fn, "TaoMlirLaunch")
        else:
            self._check_launch_op(fn, "DiscLaunch")


if __name__ == "__main__":
    unittest.main()

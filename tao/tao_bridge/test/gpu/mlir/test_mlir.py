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


from __future__ import print_function
from test.tao_ut_common import *
from test.gpu.mlir.base import MlirTestCase

import unittest
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf
import numpy as np
from google.protobuf import text_format

class TestTaoMlir(MlirTestCase):
    def setUp(self):
        MlirTestCase.setUpWithMLIR(self._testMethodName)

    def _check_launch_op(self, fn, launch_op_name):
        with open(fn) as f:
            graph_def = tf.GraphDef()
            text_format.Merge(f.read(), graph_def)

            disc_op = self.get_node(graph_def, launch_op_name)
            mlir_func_name = disc_op.attr["mlir_function"].func.name
            self.assertNotEqual(mlir_func_name, "")

            mlir_func = self.get_func(graph_def, mlir_func_name)
            self.get_node(mlir_func, "MatMul")

    def test_with_mlir(self):
        np_dtype = np.float32
        dtype = tf.float32

        def _expected_value():
            shape = [3, 3]
            x = np.ones(shape).astype(np_dtype)
            y = np.ones(shape).astype(np_dtype)
            c = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
            t1 = x + y
            t2 = np.matmul(t1, c)
            t3 = np.sum(t2)
            return t3 * t3

        g = tf.Graph()
        with g.as_default():
            shape = [3, 3]
            px = tf.placeholder(shape=shape, dtype=dtype, name="px")
            py = tf.placeholder(shape=shape, dtype=dtype, name="py")
            c = tf.constant([[1, 2, 3], [1, 2, 3], [1, 2, 3]],
                            dtype=dtype, shape=shape, name="c")
            t1 = px + py
            t2 = tf.matmul(t1, c)
            t3 = tf.reduce_sum(t2)
            t4 = t3 * t3

            with super(TestTaoMlir, self).new_sess() as sess:
                for i in range(0, 120):
                    np_x = np.ones(shape).astype(np_dtype)
                    np_y = np.ones(shape).astype(np_dtype)
                    r = sess.run(t4, {px: np_x, py: np_y})
                    self.assertEqual(r, _expected_value())

        fn = super(TestTaoMlir, self).dumped_file('after_tao_pass.pbtxt')
        if self.is_platform_alibaba():
            self._check_launch_op(fn, "TaoMlirLaunch")
        else:
            self._check_launch_op(fn, "DiscLaunch")


if __name__ == "__main__":
    unittest.main()

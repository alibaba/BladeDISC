#!/usr/bin/env python
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

    def _check_disc(self, fn):
        with open(fn) as f:
            graph_def = tf.GraphDef()
            text_format.Merge(f.read(), graph_def)

            disc_op = self.get_node(graph_def, "DiscLaunch")

            mlir_func_name = disc_op.attr["mlir_function"].func.name
            self.assertNotEqual(mlir_func_name, "")

            mlir_func = self.get_func(graph_def, mlir_func_name)
            self.get_node(mlir_func, "MatMul")

    def _check_tao(self, fn):
        with open(fn) as f:
            graph_def = tf.GraphDef()
            text_format.Merge(f.read(), graph_def)

            tao_op = self.get_node(graph_def, "TaoLaunch")

            self.assertNotEqual(tao_op.attr["function"].func.name, "")
            mlir_func_name = tao_op.attr["mlir_function"].func.name
            self.assertEqual(mlir_func_name,
                tao_op.attr["function"].func.name + "_mlir")

            mlir_func = self.get_func(graph_def, mlir_func_name)
            tao_op_inner = self.get_node(mlir_func, "TaoMlirLaunch")
            self.assertEqual(
                tao_op_inner.attr["function"].func.name, "")
            inner_mlir_func_name = tao_op_inner.attr["mlir_function"].func.name
            self.assertNotEqual(inner_mlir_func_name, "")

            inner_mlir_func = self.get_func(graph_def, inner_mlir_func_name)
            self.get_node(inner_mlir_func, "MatMul")

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
            self._check_tao(fn)
        else:
            self._check_disc(fn)


if __name__ == "__main__":
    unittest.main()

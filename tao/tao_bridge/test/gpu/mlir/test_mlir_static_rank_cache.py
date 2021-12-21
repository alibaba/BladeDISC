#!/usr/bin/env python
from __future__ import print_function
import unittest

from test.tao_ut_common import *
from test.gpu.mlir.base import MlirTestCase
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf
import numpy as np
from google.protobuf import text_format

class TestTaoMlirStaticRank0(MlirTestCase):
    def setUp(self):
        MlirTestCase.setUpWithMLIR(self._testMethodName)

    def _check_tao_time_const(self, fn):
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

            self.assertEqual(
                tao_op_inner.attr["mlir_function"].func.attr["_TaoXlaNumConstantArgs"].i, 1)
            self.assertEqual(
                tao_op_inner.attr["mlir_function"].func.attr["_TaoMlirNumFixedShapeArgs"].i, 0)
            self.assertEqual(
                tao_op.attr["function"].func.attr["_TaoXlaNumConstantArgs"].i, 1)
            self.assertEqual(
                tao_op.attr["function"].func.attr["_TaoMlirNumFixedShapeArgs"].i, 0)

    def _check_disc_time_const(self, fn):
        with open(fn) as f:
            graph_def = tf.GraphDef()
            text_format.Merge(f.read(), graph_def)

            disc_op = self.get_node(graph_def, "DiscLaunch")

            self.assertEqual(disc_op.attr["mlir_function"].func.attr["_TaoXlaNumConstantArgs"].i, 1)
            self.assertEqual(disc_op.attr["mlir_function"].func.attr["_TaoMlirNumFixedShapeArgs"].i, 0)

    # test for must_be_const inputs for mlir
    def test_compile_time_const(self):
        g = tf.Graph()
        with g.as_default():

            shape0 = [16, 16]
            in0 = np.ones(shape0)
            in1 = np.ones(shape0)
            in2 = np.array([0])
            in2_2 = np.array([1])

            with super(TestTaoMlirStaticRank0, self).new_sess() as session:
                arg1 = tf.placeholder(tf.float32, shape=shape0, name='input0')
                arg2 = tf.placeholder(tf.float32, shape=shape0, name='input2')
                dims = tf.placeholder(tf.int32, shape=[1], name='input2')
                result = tf.reduce_sum(arg1*arg2, dims)
                result_val = session.run(result, feed_dict={
                  arg1: in0,
                  arg2: in1,
                  dims: in2
                })
                result_val = session.run(result, feed_dict={
                  arg1: in0,
                  arg2: in1,
                  dims: in2_2
                })
                result_val = session.run(result, feed_dict={
                  arg1: in0,
                  arg2: in1,
                  dims: in2
                })
                expected = np.ones([16, ]) * 16
                self.assertTrue(np.allclose(result_val, expected, atol=1e-5),
                                msg="expect: {} but got: {}".format(expected, result_val))

        fname = super(TestTaoMlirStaticRank0, self).dumped_file('after_tao_pass.pbtxt')
        if self.is_platform_alibaba():
            self._check_tao_time_const(fname)
        else:
            self._check_disc_time_const(fname)

    def _check_tao_fixed_shape(self, fn):
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

            self.assertEqual(
                tao_op_inner.attr["mlir_function"].func.attr["_TaoXlaNumConstantArgs"].i, 0)
            self.assertEqual(
                tao_op_inner.attr["mlir_function"].func.attr["_TaoMlirNumFixedShapeArgs"].i, 2)
            self.assertEqual(
                tao_op.attr["function"].func.attr["_TaoXlaNumConstantArgs"].i, 2)
            self.assertEqual(
                tao_op.attr["function"].func.attr["_TaoMlirNumFixedShapeArgs"].i, 0)

    def _check_disc_fixed_shape(self, fn):
        with open(fn) as f:
            graph_def = tf.GraphDef()
            text_format.Merge(f.read(), graph_def)

            disc_op = self.get_node(graph_def, "DiscLaunch")

            self.assertEqual(disc_op.attr["mlir_function"].func.attr["_TaoXlaNumConstantArgs"].i, 2)
            self.assertEqual(disc_op.attr["mlir_function"].func.attr["_TaoMlirNumFixedShapeArgs"].i, 0)

    # test for must_be_fixed_shape inputs for mlir
    def test_compile_time_fixed_shape(self):
        g = tf.Graph()
        with g.as_default():

            shape0 = [1024, 1024]
            in0 = np.ones(shape0)
            in1 = np.ones(shape0)
            in2 = np.array([412, 1, 412])
            in3 = np.array([612, 0, 612])
            with super(TestTaoMlirStaticRank0, self).new_sess() as session:
                arg1 = tf.placeholder(tf.float32, shape=shape0, name='input0')
                arg2 = tf.placeholder(tf.float32, shape=shape0, name='input2')
                arg3 = tf.placeholder(tf.int32, shape=[None], name='input2')
                arg4 = tf.placeholder(tf.int32, shape=[None], name='input2')

                result = tf.reshape(arg1*arg2, arg3+arg4)

                result_val = session.run(result, feed_dict={
                  arg1: in0,
                  arg2: in1,
                  arg3: in2,
                  arg4: in3
                })

                in2 = np.array([412, 1, 412])
                in3 = np.array([100, 1, 612])

                result_val = session.run(result, feed_dict={
                  arg1: in0,
                  arg2: in1,
                  arg3: in2,
                  arg4: in3
                })
                expected = np.ones([512, 2, 1024])
                self.assertTrue(np.allclose(result_val, expected, atol=1e-5),
                                msg="expect: {} but got: {}".format(expected, result_val))

                in2 = np.array([412, 1, 412])
                in3 = np.array([612, 0, 612])

                result_val = session.run(result, feed_dict={
                  arg1: in0,
                  arg2: in1,
                  arg3: in2,
                  arg4: in3
                })

                expected = np.ones([1024, 1, 1024])
                self.assertTrue(np.allclose(result_val, expected, atol=1e-5),
                                msg="expect: {} but got: {}".format(expected, result_val))

        fname = super(TestTaoMlirStaticRank0, self).dumped_file('after_tao_pass.pbtxt')
        if self.is_platform_alibaba():
            self._check_tao_fixed_shape(fname)
        else:
            self._check_disc_fixed_shape(fname)


if __name__ == "__main__":
    unittest.main()

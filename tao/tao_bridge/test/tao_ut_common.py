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

import datetime as dt
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np


class TaoTestCase(unittest.TestCase):
    """The base class for all tao tests. It helps to setup and tear down test."""
    DUMP_PATH = "tmp/graph"
    LIB_TAO_OPS = None

    @staticmethod
    def _locate_tao_compiler():
        """
        Try to find tao_compiler binary under bazel build path if not specified by
        user or found already.
        """
        if 'TAO_COMPILER_PATH' not in os.environ:
            file_dir = os.path.abspath(os.path.dirname(__file__))
            compiler = os.path.join(file_dir, os.pardir, os.pardir, os.pardir, "tao_compiler",
                                    "bazel-bin", "decoupling", "tao_compiler_main")
            compiler = os.path.abspath(compiler)
            assert os.path.exists(compiler), \
                "Tao compiler not found at: " + compiler
            assert os.path.isfile(compiler), \
                "Tao compiler is not a regular file: " + compiler
            assert os.access(compiler, os.X_OK), \
                "Tao compiler is not executable: " + compiler
            os.environ['TAO_COMPILER_PATH'] = compiler

    @staticmethod
    def _locate_lib_tao_ops():
        """Try to find libtao_ops.so under tao build path."""
        import tensorflow as tf
        file_dir = os.path.abspath(os.path.dirname(__file__))
        if TaoTestCase.LIB_TAO_OPS is None:
            tao_lib = os.path.join(
                file_dir, os.pardir, os.pardir, "bazel-bin", "libtao_ops.so")
            tao_lib = os.path.abspath(tao_lib)
            assert os.path.exists(tao_lib), \
                "libtao_ops.so not found at: " + tao_lib
            assert os.path.isfile(tao_lib), \
                "libtao_ops.so is not a regular file: " + tao_lib
            TaoTestCase.LIB_TAO_OPS = tf.load_op_library(tao_lib)

    @staticmethod
    def _setup_tf_logging():
        if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ERROR log
        if 'TF_CPP_MIN_VLOG_LEVEL' not in os.environ:
            os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'

    @staticmethod
    def _setup_dump_graph(testcase=''):
        os.environ['TAO_DUMP_PASS_OUTPUT'] = 'true'
        os.environ['TAO_DUMP_WITH_UNIQUE_ID'] = 'false'
        if 'TAO_GRAPH_DUMP_PATH' not in os.environ:
            file_dir = os.path.abspath(os.path.dirname(__file__))
            TaoTestCase.DUMP_PATH = tempfile.mkdtemp(
                dir=file_dir,
                prefix="test_dump_{}_{}_".format(
                    testcase, dt.datetime.now().strftime('%m%d%H%M%S')))
            os.environ['TAO_GRAPH_DUMP_PATH'] = TaoTestCase.DUMP_PATH
            os.environ['TEST_TMPDIR'] = TaoTestCase.DUMP_PATH
        else:
            TaoTestCase.DUMP_PATH = os.environ['TAO_GRAPH_DUMP_PATH']

        if os.path.exists(TaoTestCase.DUMP_PATH):
            shutil.rmtree(TaoTestCase.DUMP_PATH)
        os.makedirs(TaoTestCase.DUMP_PATH)

    @staticmethod
    def dumped_file(name):
        """Get full path of dumped file"."""
        return os.path.join(TaoTestCase.DUMP_PATH, name)

    @staticmethod
    def setUpWithoutTaoOpLib(testcase=''):
        os.environ['BRIDGE_ENABLE_TAO'] = 'true'
        TaoTestCase._setup_dump_graph(testcase)
        TaoTestCase._setup_tf_logging()
        TaoTestCase._locate_tao_compiler()
        np.random.seed(1)

    def setUp(self):
        TaoTestCase.setUpWithoutTaoOpLib()
        TaoTestCase._locate_lib_tao_ops()

    def tearDown(self):
        if os.path.exists(TaoTestCase.DUMP_PATH) and os.getenv("KEEP_DUMP", "false") != "true":
            shutil.rmtree(TaoTestCase.DUMP_PATH)
        sys.stdout.flush()
        sys.stderr.flush()

    def new_sess(self, allow_growth=True,
                 allocator_type='BFC',
                 log_device_placement=False,
                 allow_soft_placement=True):
        try:
            import tensorflow.compat.v1 as tf
            tf.disable_v2_behavior()
        except:
            import tensorflow as tf
        gpu_options = tf.GPUOptions(
            allow_growth=True, allocator_type='BFC')
        config = tf.ConfigProto(log_device_placement=False,
                                allow_soft_placement=True,
                                gpu_options=gpu_options)
        return tf.Session(config=config)


class TaoCpuTestCase(TaoTestCase):
    """ overriding TaoTestCase setUp method to enable cpu xla jit"""

    def setUp(self):
        super(TaoCpuTestCase, self).setUp()
        os.environ['TF_XLA_FLAGS'] = "--tf_xla_cpu_global_jit=true"

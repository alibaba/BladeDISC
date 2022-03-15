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

import contextlib
import logging
import os
import sys
import unittest
from typing import Iterator, Optional

import numpy as np

try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf

from tensorflow.core.protobuf import rewriter_config_pb2

tf.disable_eager_execution()
FILE_DIR = os.path.abspath(os.path.dirname(__file__))


class TfCustomOpsTestCase(unittest.TestCase):
    """The base class for all blade tf custom op tests."""

    blade_ops = None

    @staticmethod
    def _locateLibBladeOps() -> None:
        """Try to find libtf_blade.so under build path."""
        if TfCustomOpsTestCase.blade_ops is None:
            lib_candidates = [
                os.path.join(FILE_DIR, "libtf_blade.so"),
            ]
            for blade_lib in lib_candidates:
                blade_lib = os.path.abspath(blade_lib)
                if os.path.exists(blade_lib) and os.path.isfile(blade_lib):
                    logging.info(f'loading from {blade_lib}')
                    TfCustomOpsTestCase.blade_ops = tf.load_op_library(blade_lib)
                    logging.info(f'Module {dir(TfCustomOpsTestCase.blade_ops)}')
                    return
            raise Exception(
                "libtf_blade.so not found after searching: {}".format(lib_candidates)
            )

    @staticmethod
    def _setupLogging() -> None:
        """Setup logging level."""
        if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        if "TF_CPP_MIN_VLOG_LEVEL" not in os.environ:
            os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "0"

    def setUp(self) -> None:
        TfCustomOpsTestCase._setupLogging()
        TfCustomOpsTestCase._locateLibBladeOps()
        np.random.seed(1)

    def tearDown(self) -> None:
        sys.stdout.flush()
        sys.stderr.flush()

    @contextlib.contextmanager
    def get_test_session(
        self, graph: Optional[tf.Graph] = None, config: Optional[tf.ConfigProto] = None
    ) -> Iterator[tf.Session]:
        """Returns a TF Session for use in executing tests."""

        def _getConfig(config: Optional[tf.ConfigProto]) -> tf.ConfigProto:
            if config is None:
                config = tf.ConfigProto()
                config.allow_soft_placement = True
                config.gpu_options.allow_growth = True
            config.graph_options.optimizer_options.opt_level = -1
            config.graph_options.rewrite_options.constant_folding = (
                rewriter_config_pb2.RewriterConfig.OFF
            )
            config.graph_options.rewrite_options.pin_to_host_optimization = (
                rewriter_config_pb2.RewriterConfig.OFF
            )
            return config

        with tf.Session(graph=graph, config=_getConfig(config)) as sess:
            yield sess

    def assertAllClose(
        self,
        a: Optional[tf.Tensor],
        b: Optional[tf.Tensor],
        rtol: float = 1e-6,
        atol: float = 1e-6,
        msg: str = '',
    ) -> None:
        """Asserts that two structures of np.arrays or Tensors, have near values."""

        def _getNdArray(x: Optional[tf.Tensor]) -> np.array:
            # If x is x tensor then convert it to ndarray
            if isinstance(x, tf.Tensor):
                with self.get_test_session():
                    x = x.eval()
            if not isinstance(x, np.ndarray):
                x = np.array(x)
            return x

        a = _getNdArray(a)
        b = _getNdArray(b)
        shape_mismatch_msg = "Shape mismatch: expected {}, got {}.".format(
            a.shape, b.shape
        )
        self.assertEqual(a.shape, b.shape, shape_mismatch_msg)
        msgs = [msg]
        if not np.allclose(a, b, rtol=rtol, atol=atol):
            # Adds more details to np.testing.assert_allclose.
            cond = np.logical_or(
                np.abs(a - b) > atol + rtol * np.abs(b), np.isnan(a) != np.isnan(b)
            )
            if a.ndim:
                x = a[np.where(cond)]
                y = b[np.where(cond)]
                msgs.append("not close where = {}".format(np.where(cond)))
            else:
                # np.where is broken for scalars
                x, y = a, b
            msgs.append("not close lhs = {}".format(x))
            msgs.append("not close rhs = {}".format(y))
            msgs.append("not close dif = {}".format(np.abs(x - y)))
            msgs.append("not close tol = {}".format(atol + rtol * np.abs(y)))
            msgs.append("dtype = {}, shape = {}".format(a.dtype, a.shape))
            np.testing.assert_allclose(
                a, b, rtol=rtol, atol=atol, err_msg="\n".join(msgs), equal_nan=True
            )

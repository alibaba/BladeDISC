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


from test.tao_ut_common import TaoTestCase
import unittest
import numpy as np
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    tf.config.run_functions_eagerly(False)
except:
    import tensorflow as tf
from tensorflow.python.framework.errors_impl import InvalidArgumentError



class TestDiscFakeQuant(unittest.TestCase):
    def setUp(self):
        TaoTestCase._locate_lib_tao_ops()

    def _run_fake_quant(self, axis):
        input = np.random.rand(2,3).astype(np.float32)
        scale = np.array([0.1], dtype=np.float32)
        zero_point = np.array([0], dtype=np.int32)
        with tf.Session() as sess:
            output = TaoTestCase.LIB_TAO_OPS.disc_fake_quant(
                tf.constant(input),
                tf.constant(scale),
                tf.constant(zero_point),
                quant_min=-128,
                quant_max=127,
                num_bits=8,
                axis=axis,
                use_signed=True,
                use_symmetric=True,
                use_dynamic=True)
            output = sess.run(output)
            self.assertTrue(np.allclose(output, input))

    def test_per_tensor(self):
        self._run_fake_quant(axis=[])

    def test_per_channel(self):
        self._run_fake_quant(axis=[1])


if __name__ == "__main__":
    unittest.main()


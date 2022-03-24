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

from typing import Dict, Optional

import numpy as np

from tf_blade.util.tf_import_helper import tf

tf.disable_v2_behavior()


class ConvNetMaker:
    def _conv2d(
        self, x: tf.Tensor, w: tf.Tensor, b: tf.Tensor, strides: int = 1
    ) -> tf.Tensor:
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding="SAME")
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def _maxpool2d(self, x: tf.Tensor, k: int = 2) -> tf.Tensor:
        return tf.nn.max_pool(
            x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME"
        )

    def _conv_net(
        self, x: tf.Tensor, weights: tf.Tensor, biases: tf.Tensor
    ) -> tf.Tensor:

        # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
        conv1 = self._conv2d(x, weights["wc1"], biases["bc1"])
        # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
        conv1 = self._maxpool2d(conv1, k=2)

        # Convolution Layer
        # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
        conv2 = self._conv2d(conv1, weights["wc2"], biases["bc2"])
        # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
        conv2 = self._maxpool2d(conv2, k=2)

        conv3 = self._conv2d(conv2, weights["wc3"], biases["bc3"])
        # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
        conv3 = self._maxpool2d(conv3, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv3, [-1, weights["wd1"].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights["wd1"]), biases["bd1"])
        fc1 = tf.nn.relu(fc1)
        # Output, class prediction
        # finally we multiply the fully connected layer with the weights and add a bias term.
        out = tf.add(tf.matmul(fc1, weights["out"]), biases["out"], name="out")
        return out

    def gen_simple_conv_net(self, batch: Optional[int] = 1) -> tf.GraphDef:
        tf.reset_default_graph()
        n_classes = 10
        if batch == -1:
            batch = None
        x = tf.placeholder(dtype=tf.float32, shape=[batch, 28, 28, 1], name="input")

        weights = {
            "wc1": tf.constant(np.random.rand(3, 3, 1, 32).astype(dtype=np.float32)),
            "wc2": tf.constant(np.random.rand(3, 3, 32, 64).astype(dtype=np.float32)),
            "wc3": tf.constant(np.random.rand(3, 3, 64, 128).astype(dtype=np.float32)),
            "wd1": tf.constant(
                np.random.rand(4 * 4 * 128, 128).astype(dtype=np.float32)
            ),
            "out": tf.constant(np.random.rand(128, n_classes).astype(dtype=np.float32)),
        }
        biases = {
            "bc1": tf.constant(np.random.rand(32).astype(dtype=np.float32)),
            "bc2": tf.constant(np.random.rand(64).astype(dtype=np.float32)),
            "bc3": tf.constant(np.random.rand(128).astype(dtype=np.float32)),
            "bd1": tf.constant(np.random.rand(128).astype(dtype=np.float32)),
            "out": tf.constant(np.random.rand(10).astype(dtype=np.float32)),
        }

        self._conv_net(x, weights, biases)
        graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)

        return graph_def

    def get_simple_conv_net_feed_dict(self, batch: int = 1) -> Dict:
        input_data = np.random.uniform(size=[batch, 28, 28, 1]).astype(np.float32)
        return {"input:0": input_data}

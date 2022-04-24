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

import os
import time
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.saved_model import tag_constants
# from tensorflow.python.framework import convert_to_constants
tf.disable_v2_behavior()

import blade_disc_tf as disc


def load_frozen_graph(model_file: str):
    graph_def = tf.GraphDef()
    with open(model_file, 'rb') as f:
        graph_def.ParseFromString(f.read())
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name='')
    return graph


def run_bert(optimize_config: str = None):
    if optimize_config is 'disc':
        disc.enable()

    session_config = tf.ConfigProto()
    session_config.allow_soft_placement = True
    session_config.gpu_options.allow_growth = True
    session_config.graph_options.rewrite_options.auto_mixed_precision = 1
    if optimize_config is "xla":
        session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    model_dir = "saved_model"
    sess = tf.Session(graph=tf.Graph(), config=session_config)
    tf.saved_model.loader.load(sess, [tag_constants.SERVING], model_dir)

    fetch = [
        "StatefulPartitionedCall:0", "StatefulPartitionedCall:1",
        "StatefulPartitionedCall:2", "StatefulPartitionedCall:3",
        "StatefulPartitionedCall:4"
    ]
    feed_dict = {
        'serving_default_input_1:0': np.ones((30, 10), dtype=int),
        'serving_default_input_2:0': np.zeros((1), dtype=int),
        'serving_default_input_3:0': np.ones((1), dtype=float),
        'serving_default_input_4:0': np.ones((1), dtype=float),
        'serving_default_input_5:0': np.ones((1), dtype=float),
    }


    # warmup.
    for i in range(20):
        outs = sess.run(fetch, feed_dict=feed_dict)

    # evaluate.
    iters = 100
    tic = time.time()
    for i in range(iters):
        outs = sess.run(fetch, feed_dict=feed_dict)
    avg_time = (time.time() - tic) / iters
    print("average time in {} iterations: {} seconds".format(iters, avg_time))


if __name__ == '__main__':
    # `optimize_config` can be 'xla', 'disc' or None.
    run_bert(optimize_config='disc')

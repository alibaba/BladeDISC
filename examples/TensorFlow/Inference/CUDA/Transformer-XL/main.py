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
tf.disable_v2_behavior()

import blade_disc_tf as disc


def load_frozen_graph(model_file : str):
    graph_def = tf.GraphDef()
    with open(model_file, 'rb') as f:
        graph_def.ParseFromString(f.read())
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name='')
    return graph


def run_bert(optimize_config : str = None):
    if optimize_config is 'disc':
        disc.enable()

    session_config = tf.ConfigProto()
    session_config.allow_soft_placement = True
    session_config.gpu_options.allow_growth = True
    session_config.graph_options.rewrite_options.auto_mixed_precision = 1
    if optimize_config is "xla":
        session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    # to use frozen_model
    graph = load_frozen_graph('./model/frozen_model.pb')
    sess = tf.Session(graph = graph, config = session_config)

    fetch = ["transformer/Mean:0", "transformer/StopGradient:0","Size:0"]

    # NOTE: PRINT INFO of INPUT: Tensor("IteratorGetNext:0", shape=(16, ?), dtype=int32).
    feed_dict = {
            'IteratorGetNext:0' : np.ones((16, 192), dtype=int),
            }

    outs = sess.run(fetch, feed_dict = feed_dict)
    # print(outs)


if __name__ == '__main__':
    # `optimize_config` can be 'xla', 'disc' or None.
    run_bert(optimize_config = 'disc')


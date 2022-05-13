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

import time, sys
import numpy as np
import tensorflow.compat.v1 as tf
import click

tf.disable_v2_behavior()

def load_frozen_graph(model_file: str):
    graph_def = tf.GraphDef()
    with open(model_file, 'rb') as f:
        graph_def.ParseFromString(f.read())
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name='')
    return graph

@click.command()
@click.option("--disc", is_flag=True)
@click.option("--fp16", is_flag=True)
@click.option("--xla", is_flag=True)
@click.option("--batch", type=int, default=32)
def run_bert(disc, fp16, batch, xla):
    if disc:
        sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../common"))
        import disc_init
    session_config = tf.ConfigProto()
    session_config.allow_soft_placement = True
    session_config.gpu_options.allow_growth = True
    if fp16:
        session_config.graph_options.rewrite_options.auto_mixed_precision = 1
    if xla:
        session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    graph = load_frozen_graph('./model/frozen.pb')
    sess = tf.Session(graph=graph, config=session_config)

    # Warmup.
    print("Warming up...")
    fetch = ["unstack:0", "unstack:1"]
    feed_dict = {
        'input_ids_1:0': np.ones((1, 384), dtype=int),
        'segment_ids_1:0': np.zeros((1, 384), dtype=int),
        'input_mask_1:0': np.ones((1, 384), dtype=int),
    }
    for i in range(50):
        outs = sess.run(fetch, feed_dict=feed_dict)

    # Measure performance.
    print("Run 10 inferences with dynamic batch sizes.")
    all_times = []
    if batch == -1:
        targets = [2, 2, 4, 1, 1, 8, 8, 2, 16, 2]
    else:
        targets = [batch] * 10
    for batch in targets:
        feed_dict = {
            'input_ids_1:0': np.ones((batch, 384), dtype=int),
            'segment_ids_1:0': np.zeros((batch, 384), dtype=int),
            'input_mask_1:0': np.ones((batch, 384), dtype=int),
        }
        s = time.time()
        outs = sess.run(fetch, feed_dict=feed_dict)
        e = time.time()
        print(f'inference batch-size {batch}: {e - s} s.')
        all_times.append(e - s)
    print(f'total: {np.sum(all_times)} s.')


if __name__ == '__main__':
    # `optimize_config` can be 'xla', 'disc' or None.
    run_bert()

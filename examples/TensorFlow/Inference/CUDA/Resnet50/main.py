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
import time
import numpy as np
import tensorflow.compat.v1 as tf
import click, sys
tf.disable_v2_behavior()

#import blade_disc_tf as disc


def load_frozen_graph(model_file : str):
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
    pbfile = './tf_resnet50_v1.5/frozen.pb'
    graph = load_frozen_graph(pbfile)
    sess = tf.Session(graph = graph, config = session_config)

    gf = tf.GraphDef()
    gf.ParseFromString(open(pbfile, 'rb').read())
    for n in gf.node:
        if n.name == "input_tensor":
            print(n.name, n.op, n)

    # Warmup.
    print("Warming up...")
    datafile = "tf_resnet50_v1.5/test_bc32.npy"
    data = np.load(datafile, allow_pickle=True, encoding="bytes").item()

    fetch = ["softmax_tensor:0"]
    feed_dict = data
#    print(data.values())
            
    for i in range(5):
        outs = sess.run(fetch, feed_dict = feed_dict)

    # Measure performance.
    print("Run {} inferences with dynamic batch sizes.".format(batch))
    #384
    all_times = []
    targets = [2, 2, 4, 1, 1, 8, 8, 2, 16, 2]
    targets = [32] * batch
    for _ in targets:
        s = time.time()
        outs = sess.run(fetch, feed_dict = feed_dict)
        e = time.time()
        #print(f'inference batch-size {batch}: {e - s} s.')
        all_times.append(e - s)
    print(f'total: {np.sum(all_times)} s.')


if __name__ == '__main__':
    # `optimize_config` can be 'xla', 'disc' or None.
    run_bert()

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
import shutil
import time

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from tensorflow.python.saved_model import loader

# huggingface
from transformers import TFBertModel

from tf_blade.gpu.tf_to_trt import Tf2TrtOpt
from tf_blade.common.tf_grappler import GrapplerBasicOpt


def get_tf_bert_model():
    model = TFBertModel.from_pretrained("bert-base-cased")  # Automatically loads the config
    return model


def get_tf_graph_def():
    bert_model = get_tf_bert_model()
    model_dir = 'bert_pretrained'
    if os.path.exists(model_dir) and not loader.maybe_saved_model_directory(model_dir):
        shutil.rmtree(model_dir)
        tf.saved_model.save(bert_model, model_dir)
    elif not os.path.exists(model_dir):
        tf.saved_model.save(bert_model, model_dir)
    loaded = tf.saved_model.load(model_dir)
    # using default signature
    func = loaded.signatures["serving_default"]
    fetches = [output.name for output in func.outputs]
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2,
    )
    func = convert_variables_to_constants_v2(func)
    graph_def = func.graph.as_graph_def()
    opt = GrapplerBasicOpt()
    output_graph_def = opt.optimize_graph_def(graph_def, fetches)
    return output_graph_def, fetches


def run_benchmark(graph_def, fetches, feed_dict, model_name):
    tf.compat.v1.reset_default_graph()
    session_config = tf.compat.v1.ConfigProto()
    session_config.allow_soft_placement = True
    session_config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=session_config) as sess:
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name="")
        output = sess.run(fetches, feed_dict)
        # Warmup!
        for i in range(0, 100):
            sess.run(fetches, feed_dict)

        # Benchmark!
        num_runs = 300
        start = time.time()
        for i in range(0, num_runs):
            sess.run(fetches, feed_dict)
        elapsed = time.time() - start
        rt_ms = elapsed / num_runs * 1000.0

        # Show the result!
        print("Latency of {} model: {:.2f}".format(model_name, rt_ms))
    return output


origin_graph_def, fetches = get_tf_graph_def()
feed_dicts = list()
feed_dicts.append({
    'input_ids:0' : np.ones((1, 5), dtype=int),
})

opt_pass = Tf2TrtOpt()

model_outputs = [fetch.split(":")[0] for fetch in fetches]
opt_graph_def = opt_pass.optimize_graph_def(
    origin_graph_def, model_outputs, feed_dicts, True,
)

output_origin = run_benchmark(origin_graph_def, fetches, feed_dicts[0], "origin")
output_opt = run_benchmark(opt_graph_def, fetches, feed_dicts[0], "optimized")
assert(len(output_origin) == len(output_opt))
for i in range(len(output_origin)):
    assert(np.allclose(output_origin[i], output_opt[i], rtol=1e-6, atol=1e-3))

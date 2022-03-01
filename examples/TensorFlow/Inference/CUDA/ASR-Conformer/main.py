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

#os.environ["TF_CPP_VMODULE"] = "gpu_driver=1,ral_context=1,ral_api=1,tf_context_impl=1,mlir_executable=1,cuda_context_impl=1"
#os.environ["TAO_CPP_VMODULE"] = "gpu_driver=1,ral_context=1,ral_api=1,tf_context_impl=1,mlir_executable=1,cuda_context_impl=1"
#os.environ["TAO_VERBOSE_COMPILATION_ERR_LOG"] = "true"
#os.environ["TAO_ENFORCE_VERBOSE_COMPILATION_LOG"] = "true"
#os.environ["TAO_MLIR_DUMP"] = "true"


import time
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import blade_disc_tf as disc


def run_bert(optimize_config : str = None):
    print(os.getpid())
    #input("Press Enter to continue...")

    if optimize_config is 'disc':
        disc.enable()

    session_config = tf.ConfigProto()
    session_config.allow_soft_placement = True
    session_config.gpu_options.allow_growth = True
    session_config.graph_options.rewrite_options.auto_mixed_precision = 1
    if optimize_config is "xla":
        session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    sess = tf.Session(config = session_config)
    tags = [tf.saved_model.tag_constants.SERVING]
    export_dir = 'saved_model'
    tf.saved_model.load(sess = sess, tags = tags, export_dir = export_dir)

    # Warmup.
    print("Warming up...")
    fetch = ["StatefulPartitionedCall:0"]
    feed_dict = {
            'serving_default_signal:0' : np.random.randint(256, size = (512)),
            }
    #for i in range(50):
    for i in range(1):
        outs = sess.run(fetch, feed_dict = feed_dict)

    print(outs)


if __name__ == '__main__':
    # `optimize_config` can be 'xla', 'disc' or None.
    run_bert(optimize_config = 'disc')

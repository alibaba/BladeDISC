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
import os
from test.tao_ut_common import TaoTestCase
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf
import numpy as np
from google.protobuf import text_format

class MlirTestCase(TaoTestCase):
    @staticmethod
    def setUpWithMLIR(testcase=''):
        os.environ['BRIDGE_ENABLE_TAO'] = 'true'
        os.environ['TF_ENABLE_TAO'] = 'true'
        os.environ['TAO_ENABLE_MLIR'] = 'true'
        os.environ['TAO_EXPERIMENTAL_ENABLE_MLIR_WHOLE_GRAPH_COMPILATION'] = 'true'
        os.environ["TF_XLA_FLAGS"]="--tf_xla_min_cluster_size=1"
        os.environ["TAO_COMPILATION_MODE_ASYNC"] = "false"
        os.environ["TAO_ENABLE_FALLBACK"] = "false"
        os.environ["TAO_MLIR_BRANCH_ONLY"] = "true"
        TaoTestCase._setup_dump_graph(testcase)
        TaoTestCase._setup_tf_logging()
        TaoTestCase._locate_tao_compiler()
        TaoTestCase._locate_lib_tao_ops()

    def get_node(self, graph_or_func, op):
        nodes = graph_or_func.node if isinstance(
            graph_or_func, tf.GraphDef) else graph_or_func.node_def
        ops = list(filter(lambda n: n.op == op, nodes))
        self.assertEqual(len(ops), 1)
        return ops[0]

    def get_func(self, graph, name):
        func_list = list(filter(lambda func: func.signature.name == name,
                                graph.library.function))
        self.assertEqual(len(func_list), 1)
        return func_list[0]

    def is_platform_alibaba(self):
        return "PLATFORM_ALIBABA" in os.environ

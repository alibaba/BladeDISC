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

from typing import List

from tensorflow.core.protobuf import config_pb2, rewriter_config_pb2
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.training import saver

from tf_blade.util.tf_import_helper import tf


class GrapplerBasicOpt:
    def __init__(self) -> None:
        super().__init__()

    def optimize_graph_def(
        self, graph_def: tf.GraphDef, protected_nodes: List[str]
    ) -> tf.GraphDef:
        if not tf.get_default_session():
            tf.reset_default_graph()
        # run with default setting is enough
        config = config_pb2.ConfigProto()
        config.graph_options.rewrite_options.CopyFrom(
            rewriter_config_pb2.RewriterConfig(
                # on gpu, we leave the layout transformation to be done by TRT or HIE
                # otherwise, it will cause strange problems when optimizing with TRT or HIE
                layout_optimizer=rewriter_config_pb2.RewriterConfig.OFF,
                constant_folding=rewriter_config_pb2.RewriterConfig.ON,
                shape_optimization=rewriter_config_pb2.RewriterConfig.OFF,
                remapping=rewriter_config_pb2.RewriterConfig.OFF,
                arithmetic_optimization=rewriter_config_pb2.RewriterConfig.OFF,
                # prune useless control flow ops
                dependency_optimization=rewriter_config_pb2.RewriterConfig.ON,
                loop_optimization=rewriter_config_pb2.RewriterConfig.OFF,
                function_optimization=rewriter_config_pb2.RewriterConfig.ON,
                debug_stripper=rewriter_config_pb2.RewriterConfig.OFF,
            )
        )

        graph = tf.Graph()
        with graph.as_default():
            fetches = tf.import_graph_def(
                graph_def, return_elements=protected_nodes, name=""
            )
            # Add a collection "train_op" so that Grappler knows the inputs and outputs.
            for fetch in fetches:
                graph.add_to_collection("train_op", fetch)
            metagraph = saver.export_meta_graph(graph_def=graph.as_graph_def())
            try:
                opt_graph_def = tf_optimizer.OptimizeGraph(
                    config, metagraph, verbose=False
                )
            except Exception as err:
                raise Exception(f"Failed to do grappler opt due to: {err}")
            else:
                return opt_graph_def


class GrapplerAMPOpt:
    def __init__(self) -> None:
        super().__init__()

    def optimize_graph_def(
        self, graph_def: tf.GraphDef, protected_nodes: List[str],
    ) -> tf.GraphDef:
        tf.reset_default_graph()
        # run with default setting is enough
        config = config_pb2.ConfigProto()
        config.graph_options.rewrite_options.CopyFrom(
            rewriter_config_pb2.RewriterConfig(
                auto_mixed_precision=rewriter_config_pb2.RewriterConfig.ON,
            )
        )

        graph = tf.Graph()
        with graph.as_default():
            fetches = tf.import_graph_def(
                graph_def, return_elements=protected_nodes, name=""
            )
            # Add a collection "train_op" so that Grappler knows the inputs and outputs.
            for fetch in fetches:
                graph.add_to_collection("train_op", fetch)
            metagraph = saver.export_meta_graph(graph_def=graph.as_graph_def())
            try:
                opt_graph_def = tf_optimizer.OptimizeGraph(
                    config, metagraph, verbose=False
                )
            except Exception as err:
                raise Exception(f"Failed to do grappler opt due to: {err}")
            else:
                return opt_graph_def

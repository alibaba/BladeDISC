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

'''
Author:
2.0 qiheng.fpf@alibaba-inc.com
1.0 wanchen.swc@alibaba-inc.com

Modify the graph to use BladeFusedLSTMElementWise op

'''

import logging
from typing import List, Optional, Set

from tf_blade.util import tf_graph_transform_util as graph_transform
from tf_blade.util.simple_graph import SimpleGraph, SimpleNode
from tf_blade.util.tf_import_helper import tf


class TfLstmElementWiseOps:
    def __init__(self) -> None:
        # the following will be initialized in optimize() later
        self.graph_def: tf.GraphDef
        self.simple_graph: SimpleGraph
        self.graph_inputs: List[str] = list()
        self.graph_outputs: List[str] = list()

        self.pattern_list = list()
        self.pattern_list.append(
            SimpleNode('h/Mul', 'Mul', ['c1/Tanh', 'o/Sigmoid'], ['0'])
        )
        self.pattern_list.append(SimpleNode('c1/Tanh', 'Tanh', ['c1/Add'], ['h/Mul']))
        self.pattern_list.append(
            SimpleNode('c1/Add', 'Add', ['cf/Mul', 'ci/Mul'], ['c1/Tanh', '1'])
        )
        self.pattern_list.append(
            SimpleNode('ci/Mul', 'Mul', ['i/Sigmoid', 'c/Tanh'], ['c1/Add'])
        )
        self.pattern_list.append(
            SimpleNode('i/Sigmoid', 'Sigmoid', ['Split'], ['ci/Mul'])
        )
        self.pattern_list.append(SimpleNode('c/Tanh', 'Tanh', ['Split'], ['ci/Mul']))
        self.pattern_list.append(
            SimpleNode('cf/Mul', 'Mul', ['0', 'f/Sigmoid'], ['c1/Add'])
        )
        self.pattern_list.append(
            SimpleNode('f/Sigmoid', 'Sigmoid', ['f/Add'], ['cf/Mul'])
        )
        self.pattern_list.append(
            SimpleNode('f/Add', 'Add', ['Split', '1'], ['f/Sigmoid'])
        )
        self.pattern_list.append(
            SimpleNode('o/Sigmoid', 'Sigmoid', ['Split'], ['h/Mul'])
        )
        self.pattern_list.append(
            SimpleNode(
                'Split',
                'Split',
                ['Split/dim', 'BiasAdd'],
                ['i/Sigmoid', 'c/Tanh', 'f/Add', 'o/Sigmoid'],
            )
        )
        self.pattern_list.append(SimpleNode('Split/dim', 'Const', ['*'], ['Split']))
        self.pattern_list.append(
            SimpleNode('BiasAdd', 'BiasAdd', ['2', '3'], ['Split'])
        )

    def _process_pattern(self, pattern_list: List[SimpleNode]) -> int:
        pattern = {node.name: node for node in pattern_list}
        first_key = pattern_list[0].name
        pattern_map_list = graph_transform.get_matched_pattern(
            self.simple_graph, pattern, first_key
        )
        count = 0
        if len(pattern_map_list) == 0:
            return count

        nodes_to_remove: Set = set()

        for pattern_map in pattern_map_list:
            output_h_name = pattern_map['h/Mul']
            output_c_name = pattern_map['c1/Add']
            cf_mul_name = pattern_map['cf/Mul']
            cf_mul_node = self.simple_graph.get_node_by_name(cf_mul_name)
            f_add_name = pattern_map['f/Add']
            f_add_node = self.simple_graph.get_node_by_name(f_add_name)
            bias_add_name = pattern_map['BiasAdd']
            bias_add_node = self.simple_graph.get_node_by_name(bias_add_name)
            input_c_name = cf_mul_node.input[0]
            forget_bias_name = f_add_node.input[1]
            input_x_name = bias_add_node.input[0]
            bias_offset_name = bias_add_node.input[1]
            # add fused LSTM element-wise node
            node = self.graph_def.node.add()
            node.name = '{}_lstm_elewise'.format(output_h_name)
            node.op = graph_transform.OpType.BLADE_FUSED_LSTM_ELEMENT_WISE.value
            inputs = [input_x_name, input_c_name, bias_offset_name, forget_bias_name]
            node.input.extend(inputs)
            graph_transform.copy_node_attr(bias_add_node, 'T', 'T', node)
            # Rename inputs
            data_type = graph_transform.get_node_type(node, 'T')
            graph_transform.add_identity(
                self.graph_def, '{}:0'.format(node.name), output_c_name, data_type
            )
            graph_transform.add_identity(
                self.graph_def, '{}:1'.format(node.name), output_h_name, data_type
            )
            # Remove nodes
            graph_transform.add_remove_list(
                nodes_to_remove,
                self.simple_graph,
                list(pattern_map.values()),
                True,
                pattern_map,
                self.graph_outputs,
            )
            graph_transform.add_remove_list(
                nodes_to_remove, self.simple_graph, [output_h_name, output_c_name]
            )

            count += 1

        graph_transform.remove_node_by_index(self.graph_def, nodes_to_remove)

        # Transform: strip unused nodes & sort by execution order
        self.graph_def = graph_transform.sort_by_execution_order(
            self.graph_def, self.graph_inputs, self.graph_outputs
        )

        return count

    def optimize_graph_def(
        self, graph_def: tf.GraphDef, graph_inputs: List[str], graph_outputs: List[str],
    ) -> tf.GraphDef:
        self.graph_def = graph_def
        self.simple_graph = SimpleGraph(self.graph_def)
        self.graph_inputs = graph_inputs
        self.graph_outputs = graph_outputs

        pattern_count = self._process_pattern(self.pattern_list)
        logging.info(f'Optimize {pattern_count} patterns of LSTM element wise ops')

        if pattern_count > 0:
            return self.graph_def
        return graph_def

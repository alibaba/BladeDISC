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

import unittest
from typing import Dict, List

import numpy as np

from tf_blade.util.simple_graph import SimpleNode
from tf_blade.util.tf_hierarchy_pattern_match import TfBladePatternMatch
from tf_blade.util.tf_import_helper import tf

tf.disable_v2_behavior()


class PatternLevelZero(TfBladePatternMatch):
    blade_op_type = "TestMatMul"

    def validate_matched_map(self, matched_map: Dict[str, str]) -> bool:
        return True

    def replace_valid_pattern(self, matched_map: Dict[str, str]) -> None:
        x = self.get_graph_node_by_pattern_node_name(matched_map, "matmul")
        x.op = self.blade_op_type

    def get_pattern(self) -> List[SimpleNode]:
        pattern_list = list()
        pattern_list.append(SimpleNode("matmul", "MatMul", ["0", "b"], ["0"]))
        pattern_list.append(SimpleNode("b", "Const|Enter", ["*"], ["matmul"]))
        return pattern_list


class PatternLevelOne(TfBladePatternMatch):
    blade_op_type = "TestFusedMatMulBiasAdd"

    def validate_matched_map(self, matched_map: Dict[str, str]) -> bool:
        return True

    def replace_valid_pattern(self, matched_map: Dict[str, str]) -> None:
        x = self.get_graph_node_by_pattern_node_name(matched_map, "test_matmul")
        add = self.get_graph_node_by_pattern_node_name(matched_map, "bias_add")
        add.op = self.blade_op_type
        self.update_inputs_on_whole_graph({str(x.name): str(add.name)})

    def get_pattern(self) -> List[SimpleNode]:
        pattern_list = list()
        pattern_list.append(
            SimpleNode("test_matmul", "TestMatMul", ["0", "1"], ["bias_add"])
        )
        pattern_list.append(SimpleNode("bias", "Const|Enter", ["*"], ["bias_add"]))
        pattern_list.append(
            SimpleNode("bias_add", "BiasAdd|Add", ["test_matmul", "bias"], ["matmul"])
        )
        return pattern_list


class BladePatternMatchTest(unittest.TestCase):
    def _create_graph(self) -> tf.GraphDef:
        """placeholder->matmul->biasAdd->reshape"""
        tf.reset_default_graph()
        a = tf.placeholder(dtype=tf.float32, shape=[None, 128], name="input")
        data_b = np.random.random_sample([128, 64]).astype(np.float32)
        b = tf.constant(data_b)
        c = tf.matmul(a, b, transpose_a=False, transpose_b=False)
        data_bias = np.random.random_sample([64]).astype(np.float32)
        bias = tf.constant(data_bias)
        tf.nn.bias_add(c, bias, name="output")
        graph_def = tf.get_default_graph().as_graph_def()
        tf.reset_default_graph()
        return graph_def

    def test_blade_pattern_match(self) -> None:
        graph_def = self._create_graph()
        lv0_opt = PatternLevelZero(graph_def)
        lv0_gd = lv0_opt.get_optimized_graph_def()
        lv1_opt = PatternLevelOne(lv0_gd)
        lv1_gd = lv1_opt.get_optimized_graph_def()
        valid_lv0_opt = False
        valid_lv1_opt = False
        assert lv1_gd is not None
        for nd in lv1_gd.node:
            if nd.op == "TestMatMul":
                valid_lv0_opt = True
            if nd.op == "TestFusedMatMulBiasAdd":
                valid_lv1_opt = True
        self.assertTrue(valid_lv0_opt)
        self.assertTrue(valid_lv1_opt)


if __name__ == "__main__":
    unittest.main()

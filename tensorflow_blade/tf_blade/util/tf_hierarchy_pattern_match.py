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

import logging
from typing import Dict, List, Optional, Set

from tf_blade.util.simple_graph import SimpleGraph, SimpleNode
from tf_blade.util.tf_graph_transform_util import (
    get_matched_pattern,
    get_node_by_name,
    remove_node_by_index,
    rename_node_inputs,
)
from tf_blade.util.tf_import_helper import tf


# Basic Class for Pattern Match on tf.v1.compat.GraphDef
class TfBladePatternMatch:
    tf_blade_op_type: str = ''

    def __init__(self, graph_def: tf.GraphDef):
        self.graph_def = graph_def
        # count optimized pattern number
        self.count_opt = 0
        # {key: modified_node_name, value: before_modified_node_def } This dictionary is maintained for optimization revert
        self.opt_map: Dict[str, tf.NodeDef] = {}
        # List of SimpleGraph Nodes describes the Pattern
        self.pattern_node_list = self.get_pattern()
        self.pattern_root: Optional[SimpleNode] = None
        self.simple_graph = SimpleGraph(self.graph_def)

    def validate_matched_map(self, matched_map: Dict[str, str]) -> bool:
        """Overrides in Derived Class
        returns True or False
        Checking whether a matched pattern is valid or not,
          e.g: check some node attr[T].type, if it matches the expected data type.
        """
        return False

    def replace_valid_pattern(self, matched_map: Dict[str, str]) -> None:
        """Overrides in Derived Class
        Users are allowed to perform:
            "Pattern Replace", "Matched Node Modifications" or "Add Node"
        """
        return None

    def get_pattern(self) -> Optional[List[SimpleNode]]:
        """Overrides in Derived Class
        Returns a list of the pattern, where element of the list is a SimpleNode object.
        NOTE: now we only support pattern with one unique root
        """
        return None

    # Pattern Match Trigger Function
    def get_optimized_graph_def(self) -> Optional[tf.GraphDef]:
        self._optimize_pattern()
        if self.count_opt > 0:
            logging.info(
                "Optimized {} [{}] pattern.".format(
                    self.count_opt, self.tf_blade_op_type
                )
            )
            self._remove_optimization_replaced_nodes()
            return self.graph_def
        logging.info(
            "None Optimization for [{}] pattern.".format(self.tf_blade_op_type)
        )
        return None

    def get_patterh_node_by_name(self, inp_name: str) -> Optional[SimpleNode]:
        inp_name = inp_name.strip()
        assert isinstance(self.pattern_node_list, list)
        assert len(self.pattern_node_list) > 0
        for p_node in self.pattern_node_list:
            if (p_node.name).strip() == inp_name:
                return p_node
        err_msg = 'Invalid Pattern Node Name {}, which do not exist in Pattern List {}.'.format(
            inp_name, [nd.name for nd in self.pattern_node_list]
        )
        logging.critical(err_msg)
        return None

    def update_inputs_on_whole_graph(self, update_dict: Dict[str, str]) -> None:
        rename_node_inputs(self.graph_def, update_dict)

    def get_pattern_dependency_inputs(
        self, matched_map: Dict[str, str]
    ) -> Optional[List[str]]:
        dep_inputs = set()
        for na in matched_map.values():
            node = get_node_by_name(self.graph_def, self.simple_graph, na)
            if len(node.input) == 0:
                continue
            for inp in node.input:
                if '^' in str(inp):
                    dep_inputs.add(inp)
        if len(dep_inputs) > 0:
            return list(dep_inputs)
        return None

    def get_graph_node_by_pattern_node_name(
        self, matched_map: Dict[str, str], pattern_node_name: str
    ) -> tf.NodeDef:
        pattern_node_name = pattern_node_name.strip()
        if pattern_node_name not in matched_map.keys():
            err_msg = 'Invalid Pattern Node Name {}, which do not exist in Pattern List {}.'.format(
                pattern_node_name, matched_map.keys()
            )
            logging.critical(err_msg)
        node_name = matched_map[pattern_node_name]
        return get_node_by_name(self.graph_def, self.simple_graph, node_name)

    def revert_optimization(self, graph_def: tf.GraphDef) -> None:
        remove_name_list: List[str] = []
        add_node_list: List[tf.NodeDef] = []
        for node in graph_def.node:
            if node.op == self.tf_blade_op_type:
                add_node_list = add_node_list + self.opt_map[str(node.name)]
                remove_name_list.append(node.name)
        if len(remove_name_list) == 0:
            logging.info(
                "No Optimized [{}] found, ignore Revert Optimization.".format(
                    self.tf_blade_op_type
                )
            )
            return
        remove_idx_set = set()
        for idx, node in enumerate(self.graph_def.node):
            if node.name in remove_name_list:
                remove_idx_set.add(idx)
        remove_node_by_index(graph_def, remove_idx_set)
        for node in add_node_list:
            new_node = graph_def.node.add()
            new_node.MergeFrom(node)
        logging.info(
            "Revert Optimization: remove {} optimizations, add back {} Nodes.".format(
                len(remove_name_list), len(add_node_list)
            )
        )
        # update self.simple_graph, since we made some changes
        self.simple_graph = SimpleGraph(self.graph_def)

    def add_into_optimization_map(
        self, new_node_name: str, remove_node_list: List[tf.NodeDef]
    ) -> None:
        copy_list = []
        for node in remove_node_list:
            copy_node = tf.NodeDef()
            copy_node.MergeFrom(node)
            copy_list.append(copy_node)
        """maintain the optimized map:
        where {key: replaced node, value: removed nodes list},
        which could help to revert the optimization.
        """
        self.opt_map[new_node_name] = copy_list

    # recursively tranverse the pattern from the input root node
    def _pattern_traversal(self, p_node: SimpleNode, visited: Set[str] = set()) -> None:
        visited.add(p_node.name)
        if len(p_node.inputs) > 0:
            for inp_name in p_node.inputs:
                if inp_name.strip() == '*':
                    continue
                if (inp_name.strip()).isdigit():
                    continue
                inp = self.get_patterh_node_by_name(inp_name)
                if inp is None:
                    logging.critical('Failed to tranverse through pattern')
                    return None
                self._pattern_traversal(inp, visited)

    # validate user provided pattern and set the unique root node
    def _validate_pattern_and_set_root(self) -> None:
        assert isinstance(self.pattern_node_list, list)
        assert len(self.pattern_node_list) > 0
        self.pattern_root = None  # clean pattern_root first
        for p_node in self.pattern_node_list:
            if not isinstance(p_node, SimpleNode):
                err_msg = 'Invalid pre-defined Pattern, node named {} is not a SimpleNode.'.format(
                    p_node.name
                )
                logging.critical(err_msg)
                return
        for idx, p_node in enumerate(self.pattern_node_list):
            visited: Set[str] = set()
            self._pattern_traversal(p_node, visited)
            if len(visited) == len(self.pattern_node_list):
                self.pattern_root = self.pattern_node_list[idx]
                break
        if not isinstance(self.pattern_root, SimpleNode):
            err_msg = 'Failed to get the root of the pre-defined Pattern, \
                    please check the Pattern Node List.'
            logging.critical(err_msg)
        return

    def _get_pattern_matched_map_list(self) -> Optional[List[Dict[str, str]]]:
        # Pattern Match
        assert isinstance(self.pattern_node_list, list)
        assert len(self.pattern_node_list) > 0
        self._validate_pattern_and_set_root()
        assert isinstance(self.pattern_root, SimpleNode)
        pattern = {node.name: node for node in self.pattern_node_list}
        pattern_map_list = get_matched_pattern(
            self.simple_graph, pattern, self.pattern_root.name
        )
        if (pattern_map_list is None) or (len(pattern_map_list) == 0):
            return None
        return pattern_map_list

    def _optimize_pattern(self) -> None:
        matched_map_list = self._get_pattern_matched_map_list()
        if matched_map_list is None:
            logging.info(
                "No candidate pattern for [{}] was found.".format(self.tf_blade_op_type)
            )
            return
        logging.info(
            "Found {} candidate pattern for [{}].".format(
                len(matched_map_list), self.tf_blade_op_type
            )
        )
        for matched_map in matched_map_list:
            """
            For each Matched Pattern Map, 2 more steps are needed:
                1. Validate Matched Pattern Map
                2. Replace Pattern with new node/nodes
            """
            if not self.validate_matched_map(matched_map):
                continue
            self.replace_valid_pattern(matched_map)
            self.count_opt = self.count_opt + 1

    def _remove_optimization_replaced_nodes(self) -> None:
        """Remove Nodes Those are designed to be replaced by the Optimization Pass.
        While do notice that:
          sometimes we do node replacement with the same name,
          so we need a kept set to keep them.
        """
        kept_set = set()
        delete_set = set()
        for k, v in self.opt_map.items():
            kept_set.add(k)
            for node_def in v:
                delete_set.add(node_def.name)
        remove_set = delete_set.difference(kept_set)
        if len(remove_set) == 0:
            return
        remove_idx_set = set()
        for idx, node in enumerate(self.graph_def.node):
            if node.name in remove_set:
                remove_idx_set.add(idx)
        remove_node_by_index(self.graph_def, remove_idx_set)
        # update self.simple_graph, since we made some changes
        self.simple_graph = SimpleGraph(self.graph_def)

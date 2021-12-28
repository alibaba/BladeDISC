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

import unittest
import torch_blade.algorithm as algorithm


class TestAlgorithm(unittest.TestCase):

    def test_union_set(self,):
        union_set = algorithm.UnionSet(6)
        # Undigraph
        #     0     2
        #  1 -- 4  / \
        #         3 - 5
        edges = {(1, 4), (3, 2), (2, 5), (5, 3)}

        has_cycle = False
        for e1, e2 in edges:
            pid_e1 = union_set.find(e1)
            pid_e2 = union_set.find(e2)
            if (pid_e1 == pid_e2):
                has_cycle = True
            union_set.union(e1, e2)

        groups = union_set.get_groups()
        self.assertTrue(has_cycle)
        self.assertEqual(len(groups), 3)
        self.assertEqual([0], groups[0])
        self.assertEqual([1, 4], groups[1])
        self.assertEqual([2, 3, 5], groups[2])

    def test_nx_dag_no_cycle(self):
        edges = {(2, 4), (3, 4), (3, 5), (5, 0), (0, 2)}
        nx_dag = algorithm.NxGraph()
        nx_dag.add_node(1)
        for e1, e2 in edges:
            nx_dag.add_edge(e1, e2)

        topolist = nx_dag.lexical_order_topolist()
        self.assertEqual(topolist, [1, 3, 5, 0, 2, 4])
        self.assertFalse(nx_dag.has_cycle())
        self.assertTrue(nx_dag.has_path(5, 0))
        self.assertFalse(nx_dag.has_path(0, 5))
        self.assertTrue(nx_dag.has_path(0, 4))
        self.assertFalse(nx_dag.has_path(1, 5))
        self.assertTrue(nx_dag.has_path(3, 0))
        self.assertFalse(nx_dag.has_path(3, 1))

    def test_nx_dag_cycle(self):
        edges = {(2, 4), (3, 4), (4, 5), (3, 5), (5, 0), (0, 2)}
        nx_dag = algorithm.NxGraph()
        nx_dag.add_node(1)
        for e1, e2 in edges:
            nx_dag.add_edge(e1, e2)

        self.assertTrue(nx_dag.has_cycle())
        self.assertTrue(nx_dag.has_path(0, 5))
        self.assertTrue(nx_dag.has_path(0, 4))
        self.assertFalse(nx_dag.has_path(1, 5))
        self.assertTrue(nx_dag.has_path(3, 0))
        self.assertFalse(nx_dag.has_path(5, 3))

    def test_adj_dag_no_cycle(self):
        edges = {(1, 2), (2, 4), (3, 4), (3, 5), (5, 0), (0, 2)}
        adj_dag = algorithm.AdjGraph(6)
        for e1, e2 in edges:
            adj_dag.add_edge(e1, e2)

        topolist = adj_dag.lexical_order_topolist()
        self.assertEqual(topolist, [1, 3, 5, 0, 2, 4])
        self.assertFalse(adj_dag.has_cycle())
        self.assertFalse(adj_dag.has_path_dfs(0, 5))
        self.assertTrue(adj_dag.has_path_dfs(0, 4))
        self.assertFalse(adj_dag.has_path_dfs(1, 5))
        self.assertTrue(adj_dag.has_path_dfs(3, 0))
        self.assertFalse(adj_dag.has_path_dfs(3, 1))

    def test_adj_dag_cycle(self):
        edges = {(1, 2), (2, 4), (3, 4), (4, 5), (3, 5), (5, 0), (0, 2)}
        adj_dag = algorithm.AdjGraph(6)
        for e1, e2 in edges:
            adj_dag.add_edge(e1, e2)

        self.assertTrue(adj_dag.has_cycle())
        self.assertTrue(adj_dag.has_path_dfs(0, 5))
        self.assertTrue(adj_dag.has_path_dfs(0, 4))
        self.assertTrue(adj_dag.has_path_dfs(1, 5))
        self.assertTrue(adj_dag.has_path_dfs(3, 0))
        self.assertFalse(adj_dag.has_path_dfs(3, 1))


if __name__ == '__main__':
    unittest.main()

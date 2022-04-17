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
from typing import List

import numpy as np
from tensorflow.python.framework import function

from tf_blade.util.simple_graph import (
    GraphDefPartitioner,
    GraphSegment,
    SimpleGraph,
    SimpleNode,
)
from tf_blade.util.tf_import_helper import tf

# In TF2, control flow are represented as functions, which make sample graphs in this
# test varies with tf version. Just fix to TF1 behavior for now.
tf.disable_v2_behavior()


class SimpleNodeTest(unittest.TestCase):
    def _create_node(self) -> SimpleNode:
        return SimpleNode(
            name="simple_name",
            op="simple_op",
            inputs=["in_a:0", "in_b:1"],
            output_nodes=["out1", "out2", "out3"],
            tensors={"simple_name:0": ["out1"], "simple_name:1": ["out2", "out3"]},
        )

    def test_simple_node(self) -> None:
        node = self._create_node()
        self.assertEqual(node.num_inputs, 2)
        self.assertEqual(node.num_outputs, 3)
        self.assertEqual(node.num_tensors, 2)
        self.assertListEqual(node.input_nodes, ["in_a", "in_b"])

    def test_equality(self) -> None:
        node1 = self._create_node()
        node2 = self._create_node()
        self.assertEqual(node1, node2)


def _create_graph_def() -> tf.GraphDef:
    """ px -> Add --> Mul -> Relu -> ABS -> Add -> Sqrt
            /     \_/                     /     \   # noqa: W605
        py -                           c -       -> Relu
    """
    g = tf.Graph()
    dtype = np.float32
    shape = [3, 3]
    with g.as_default():
        px = tf.placeholder(shape=shape, dtype=dtype, name="px")
        py = tf.placeholder(shape=shape, dtype=dtype, name="py")
        c = tf.constant(np.random.randn(*shape).astype(dtype), name="c")
        x = px + py
        x = x * x
        x = tf.nn.relu(x)
        x = tf.abs(x)
        x = x + c
        y = tf.rsqrt(x)  # noqa: F841
        z = tf.nn.relu(x)  # noqa: F841
        return g.as_graph_def()


def _create_graph_def_circle() -> tf.GraphDef:
    """                  -> Relu --
                        /          \    # noqa: W605
        px -> Add -> Mul             Mul -> Relu
            /      /    \          /
        py -    c -      -> ABS --
    """
    g = tf.Graph()
    dtype = np.float32
    shape = [3, 3]
    with g.as_default():
        px = tf.placeholder(shape=shape, dtype=dtype, name="px")
        py = tf.placeholder(shape=shape, dtype=dtype, name="py")
        c = tf.constant(np.random.randn(*shape).astype(dtype), name="c")
        x = px + py
        x = x * c
        y = tf.abs(x)
        z = tf.nn.relu(x)
        x = y * z
        x = tf.nn.relu(x)
        return g.as_graph_def()


def _create_graph_def_front_circle() -> tf.GraphDef:
    """                         -> Relu -
                               /         \  # noqa: W605
        px -> Add -> Mul -> ABS           Mul -> Relu
            /      /           \         /
        py -      c             -> Sqrt -
    """
    g = tf.Graph()
    dtype = np.float32
    shape = [3, 3]
    with g.as_default():
        px = tf.placeholder(shape=shape, dtype=dtype, name="px")
        py = tf.placeholder(shape=shape, dtype=dtype, name="py")
        c = tf.constant(np.random.randn(*shape).astype(dtype), name="c")
        x = px + py
        x = x * c
        x = tf.abs(x)
        y = tf.nn.relu(x)
        z = tf.rsqrt(x)
        x = y * z
        x = tf.nn.relu(x)
        return g.as_graph_def()


def _create_graph_def_while_loop() -> tf.GraphDef:
    """              ------------------
       px -> Mul -> | -> Add -> Mul -> | -> Relu -> Sqrt
           /        |  /      /        |
       py -    c -> | --------         |
                    |                  |
                    | i -> Add -> i    |
                    |    /             |
                    | 1 -              |
                     ------------------
    """
    g = tf.Graph()
    dtype = np.float32
    shape = [3, 3]
    with g.as_default():
        px = tf.placeholder(shape=shape, dtype=dtype, name="px")
        py = tf.placeholder(shape=shape, dtype=dtype, name="py")
        c = tf.constant(np.random.randn(*shape).astype(dtype), name="c")
        x = px * py

        i = tf.constant(1, dtype=tf.int32)
        n = tf.placeholder(shape=[], dtype=np.int32, name="n")

        def body(i, n, x):  # type: ignore
            i = i + 1
            x = x + c
            x = x * c
            return i, n, x

        _, _, x = tf.while_loop(lambda i, n, x: i < n, body, [i, n, x])

        x = tf.nn.relu(x)
        x = tf.rsqrt(x)
        return g.as_graph_def()


def _create_graph_def_with_quirky_type_attr() -> tf.GraphDef:
    """ px -> Add --> Mul -> Relu -> ABS -> Add -> Sqrt
            /     \_/                     /     \   # noqa: W605
        py -                           c -       -> Relu
    """
    g = tf.Graph()
    dtype = np.float32
    shape = [3, 3]
    with g.as_default():
        px = tf.placeholder(shape=shape, dtype=dtype, name="px")
        py = tf.placeholder(shape=shape, dtype=dtype, name="py")
        c = tf.constant(np.random.randn(*shape).astype(dtype), name="c")
        x = px + py
        x = x * x
        x = tf.nn.relu(x)
        x = tf.abs(x)
        x = x + c
        y = tf.rsqrt(x)  # noqa: F841
        z = tf.nn.relu(x)  # noqa: F841
        z = tf.cast(z, np.float64)
        return g.as_graph_def()


def _create_graph_def_ctl_depenency() -> tf.GraphDef:
    """
      Placeholder|px ──────► Abs|y1 ─────────────────┐
                  │         .                        │
                  │         .ctl_dep                 ▼
                  │         .                      Add|out
                  │         .                        ▲
                  │         ▼                        │
                  └──────► Relu|t0 ────► Relu|t1 ────┘
    """
    g = tf.Graph()
    shape = [3, 3]
    dtype = np.float32
    with g.as_default():
        px = tf.placeholder(shape=shape, dtype=dtype, name="px")
        y1 = tf.abs(px, name="y1")
        with tf.control_dependencies([y1]):
            t0 = tf.nn.relu(px, name="t0")
        t1 = tf.nn.relu(t0, name="t1")
        _ = tf.add(t1, y1, name="out")
        return g.as_graph_def()


class SimpleGraphTest(unittest.TestCase):
    def test_basic(self) -> None:
        g = SimpleGraph(_create_graph_def())
        self.assertEqual(g.num_nodes, 10)
        self.assertListEqual(g.output_nodes(), ["Rsqrt", "Relu_1"])

        expected_inputs: List[List[int]] = [
            [],  # node 0
            [],  # node 1
            [],  # node 2
            [0, 1],  # node 3
            [3, 3],  # node 4
            [4],  # node 5
            [5],  # node 6
            [6, 2],  # node 7
            [7],  # node 8
            [7],  # node 9
        ]
        for i in range(g.num_nodes):
            self.assertListEqual(g.input_nodes_index(i), expected_inputs[i])

        # Check nodes
        self.assertEqual(
            g.node(0), SimpleNode("px", "Placeholder", [], ["add"], {"px:0": ["add"]})
        )
        self.assertEqual(
            g.node(1), SimpleNode("py", "Placeholder", [], ["add"], {"py:0": ["add"]})
        )
        self.assertEqual(
            g.node(2), SimpleNode("c", "Const", [], ["add_1"], {"c:0": ["add_1"]})
        )
        self.assertEqual(
            g.node(3),
            SimpleNode(
                "add",
                "AddV2",
                ["px:0", "py:0"],
                ["mul", "mul"],
                {"add:0": ["mul", "mul"]},
            ),
        )
        self.assertEqual(
            g.node(4),
            SimpleNode("mul", "Mul", ["add:0", "add:0"], ["Relu"], {"mul:0": ["Relu"]}),
        )
        self.assertEqual(
            g.node(5),
            SimpleNode("Relu", "Relu", ["mul:0"], ["Abs"], {"Relu:0": ["Abs"]}),
        )
        self.assertEqual(
            g.node(6),
            SimpleNode("Abs", "Abs", ["Relu:0"], ["add_1"], {"Abs:0": ["add_1"]}),
        )
        self.assertEqual(
            g.node(7),
            SimpleNode(
                "add_1",
                "AddV2",
                ["Abs:0", "c:0"],
                ["Rsqrt", "Relu_1"],
                {"add_1:0": ["Rsqrt", "Relu_1"]},
            ),
        )
        self.assertEqual(g.node(8), SimpleNode("Rsqrt", "Rsqrt", ["add_1:0"], [], {}))
        self.assertEqual(g.node(9), SimpleNode("Relu_1", "Relu", ["add_1:0"], [], {}))

    def test_sort(self) -> None:
        g = SimpleGraph(_create_graph_def())
        self.assertListEqual(g.topological_sort(), [2, 1, 0, 3, 4, 5, 6, 7, 9, 8])
        self.assertListEqual(
            g.topological_sort(reverse=True), [8, 9, 7, 6, 5, 4, 3, 0, 1, 2]
        )

    def test_reachable(self) -> None:
        g = SimpleGraph(_create_graph_def())
        self.assertTrue(g.is_reachable(0, set([6, 7, 8])))
        self.assertTrue(g.is_reachable(1, set([6, 7, 8])))
        self.assertTrue(g.is_reachable(1, set([6, 7, 2])))
        self.assertFalse(g.is_reachable(1, set([2])))
        self.assertTrue(g.is_reachable(1, set([8])))
        self.assertTrue(g.is_reachable(1, set([9])))


class GraphSegmentTest(unittest.TestCase):
    def test_graph_seg(self) -> None:
        g = SimpleGraph(_create_graph_def())
        seg = GraphSegment(g, set([5, 6, 7, 8, 9]))
        self.assertListEqual(seg.output_nodes(), ["Rsqrt", "Relu_1"])
        self.assertListEqual(seg.output_offsets(), [0, 1, 2])
        self.assertListEqual(seg.input_tensors(), ["mul:0", "c:0"])

        seg = GraphSegment(g, set([5, 6, 7, 8]))
        self.assertListEqual(seg.output_nodes(), ["Rsqrt", "add_1"])
        self.assertListEqual(seg.output_offsets(), [0, 1, 2])
        self.assertListEqual(seg.input_tensors(), ["mul:0", "c:0"])

        seg = GraphSegment(g, set([0, 1]))
        self.assertListEqual(seg.output_nodes(), ["px", "py"])
        self.assertListEqual(seg.output_offsets(), [0, 1, 2])
        self.assertListEqual(seg.input_tensors(), [])

    def test_segment_graph(self) -> None:
        """ px -> Add --> Mul -> Relu -> *ABS* -> Add -> Sqrt
                /     \_/                       /     \ # noqa: W605
            py -                             c -       -> Relu
        """
        p = GraphDefPartitioner(_create_graph_def())
        seg_list = p.graph_segment_list
        self.assertEqual(len(seg_list), 2)
        self.assertListEqual(seg_list[0].node_names, ["Rsqrt", "c", "add_1"])
        self.assertListEqual(seg_list[1].node_names, ["add", "mul", "Relu"])

    def test_segment_graph_circle(self) -> None:
        """                   -> Relu --
                             /          \   # noqa: W605
            px -> Add -> Mul             Mul -> Relu
                /      /     \          /
            py -    c -       -> *ABS* -
        """
        p = GraphDefPartitioner(_create_graph_def_circle())
        seg_list = p.graph_segment_list
        self.assertEqual(len(seg_list), 2)
        self.assertListEqual(seg_list[0].node_names, ["Relu_1", "Relu", "mul_1"])
        self.assertListEqual(seg_list[1].node_names, ["c", "add", "mul"])

    def test_segment_graph_front_circle(self) -> None:
        """                           -> Relu -
                                     /         \    # noqa: W605
            px -> Add -> Mul -> *ABS*           Mul -> Relu
                /      /             \         /
            py -      c               -> Sqrt -
        """
        p = GraphDefPartitioner(_create_graph_def_front_circle())
        seg_list = p.graph_segment_list
        self.assertEqual(len(seg_list), 2)
        self.assertListEqual(
            seg_list[0].node_names, ["mul_1", "Relu_1", "Relu", "Rsqrt"]
        )
        self.assertListEqual(seg_list[1].node_names, ["c", "add", "mul"])

    def test_segment_graph_front_circle_stop_at_0(self) -> None:
        """                               \   -> Relu -
                                           \ /         \    # noqa: W605
            px -> Add -> Mul -> *ABS*       \           Mul -> Relu
                /      /             \       \         /
            py -      c               -> Sqrt \       -
        """
        p = GraphDefPartitioner(
            _create_graph_def_front_circle(), minimum_segment_size=1, outputs=["Rsqrt"]
        )
        seg_list = p.graph_segment_list
        self.assertEqual(len(seg_list), 2)
        self.assertListEqual(seg_list[0].node_names, ["Rsqrt"])
        self.assertListEqual(seg_list[1].node_names, ["c", "add", "mul"])

    def test_segment_graph_front_circle_stop_at_1(self) -> None:
        """                           -> Relu | -
                                     /        |  \  # noqa: W605
            px -> Add -> Mul -> *ABS*         |   Mul -> Relu
                /      /             \        |  /
            py -      c               -> Sqrt | -
        """
        p = GraphDefPartitioner(
            _create_graph_def_front_circle(),
            minimum_segment_size=1,
            outputs=["Rsqrt", "Relu"],
        )
        seg_list = p.graph_segment_list
        self.assertEqual(len(seg_list), 3)
        self.assertListEqual(seg_list[0].node_names, ["Relu"])
        self.assertListEqual(seg_list[1].node_names, ["Rsqrt"])
        self.assertListEqual(seg_list[2].node_names, ["c", "add", "mul"])


class SegmentToSubgraphTest(unittest.TestCase):
    def test_segment_to_subgraph(self) -> None:
        """ px -> Add --> Mul -> Relu -> *ABS* -> Add -> Sqrt
                /     \_/                       /     \ # noqa: W605
            py -                             c -       -> Relu

            Main:
            px -> Subgraph_1 -> *ABS* -> Subgraph_0 -> Sqrt(Identity)
                /                                   \   # noqa: W605
            py -                                     -> Relu

            Subgraph_1:
            Placeholder -> Add --> Mul -> Relu -> Identity
                         /     \_/
            Placeholder -

            Subgraph_0:
            Placeholder -> Add -> Sqrt -> Identity
                         /     \    # noqa: W605
                      c -       -> Identity
        """
        p = GraphDefPartitioner(_create_graph_def())
        (
            main_graph,
            subgraphs,
            ori_input_names,
            new_input_names,
            _,
        ) = p.generate_subgraph_from_segment()
        self.assertEqual(len(subgraphs), 2)
        self.assertListEqual(ori_input_names, [["Abs:0"], ["px:0", "py:0"]])
        self.assertListEqual(
            new_input_names,
            [
                ["subgraph_0_placeholder_0"],
                ["subgraph_1_placeholder_0", "subgraph_1_placeholder_1"],
            ],
        )

        m = SimpleGraph(main_graph)
        self.assertEqual(m.num_nodes, 7)
        self.assertEqual(
            m.node(0),
            SimpleNode(
                "subgraph_0",
                "",
                ["Abs:0"],
                ["Rsqrt", "Relu_1"],
                {"subgraph_0:0": ["Rsqrt"], "subgraph_0:1": ["Relu_1"]},
            ),
        )
        self.assertEqual(
            m.node(1), SimpleNode("Rsqrt", "Identity", ["subgraph_0:0"], [], {},),
        )
        self.assertEqual(
            m.node(2),
            SimpleNode(
                "subgraph_1", "", ["px:0", "py:0"], ["Abs"], {"subgraph_1:0": ["Abs"]},
            ),
        )
        self.assertEqual(
            m.node(3),
            SimpleNode(
                "px", "Placeholder", [], ["subgraph_1"], {"px:0": ["subgraph_1"]},
            ),
        )
        self.assertEqual(
            m.node(4),
            SimpleNode(
                "py", "Placeholder", [], ["subgraph_1"], {"py:0": ["subgraph_1"]},
            ),
        )
        self.assertEqual(
            m.node(5),
            SimpleNode(
                "Abs",
                "Abs",
                ["subgraph_1:0"],
                ["subgraph_0"],
                {"Abs:0": ["subgraph_0"]},
            ),
        )
        self.assertEqual(
            m.node(6), SimpleNode("Relu_1", "Relu", ["subgraph_0:1"], [], {},),
        )

        s0 = SimpleGraph(subgraphs[0])
        self.assertEqual(s0.num_nodes, 6)
        self.assertEqual(
            s0.node(0),
            SimpleNode(
                "subgraph_0_placeholder_0",
                "Placeholder",
                [],
                ["add_1"],
                {"subgraph_0_placeholder_0:0": ["add_1"]},
            ),
        )
        self.assertEqual(
            s0.node(1), SimpleNode("c", "Const", [], ["add_1"], {"c:0": ["add_1"]},),
        )
        self.assertEqual(
            s0.node(2),
            SimpleNode(
                "add_1",
                "AddV2",
                ["subgraph_0_placeholder_0:0", "c:0"],
                ["Rsqrt", "subgraph_0-add_1-0"],
                {"add_1:0": ["Rsqrt", "subgraph_0-add_1-0"]},
            ),
        )
        self.assertEqual(
            s0.node(3),
            SimpleNode(
                "Rsqrt",
                "Rsqrt",
                ["add_1:0"],
                ["subgraph_0-Rsqrt-0"],
                {"Rsqrt:0": ["subgraph_0-Rsqrt-0"]},
            ),
        )
        self.assertEqual(
            s0.node(4),
            SimpleNode("subgraph_0-Rsqrt-0", "Identity", ["Rsqrt:0"], [], {},),
        )
        self.assertEqual(
            s0.node(5),
            SimpleNode("subgraph_0-add_1-0", "Identity", ["add_1:0"], [], {},),
        )

        s1 = SimpleGraph(subgraphs[1])
        self.assertEqual(s1.num_nodes, 6)
        self.assertEqual(
            s1.node(0),
            SimpleNode(
                "subgraph_1_placeholder_0",
                "Placeholder",
                [],
                ["add"],
                {"subgraph_1_placeholder_0:0": ["add"]},
            ),
        )
        self.assertEqual(
            s1.node(1),
            SimpleNode(
                "subgraph_1_placeholder_1",
                "Placeholder",
                [],
                ["add"],
                {"subgraph_1_placeholder_1:0": ["add"]},
            ),
        )
        self.assertEqual(
            s1.node(2),
            SimpleNode(
                "add",
                "AddV2",
                ["subgraph_1_placeholder_0:0", "subgraph_1_placeholder_1:0"],
                ["mul", "mul"],
                {"add:0": ["mul", "mul"]},
            ),
        )
        self.assertEqual(
            s1.node(3),
            SimpleNode(
                "mul", "Mul", ["add:0", "add:0"], ["Relu"], {"mul:0": ["Relu"]},
            ),
        )
        self.assertEqual(
            s1.node(4),
            SimpleNode(
                "Relu",
                "Relu",
                ["mul:0"],
                ["subgraph_1-Relu-0"],
                {"Relu:0": ["subgraph_1-Relu-0"]},
            ),
        )
        self.assertEqual(
            s1.node(5),
            SimpleNode("subgraph_1-Relu-0", "Identity", ["Relu:0"], [], {},),
        )

    def test_segment_to_subgraph_front_circle(self) -> None:
        """                           -> Relu -
                                     /         \    # noqa: W605
            px -> Add -> Mul -> *ABS*           Mul -> Relu
                /      /             \         /
            py -      c               -> Sqrt -

            Main:
            px -> Subgraph_1 -> *ABS* -> Subgraph_0 -> Relu(Identity)
                /
            py -

            Subgraph_1:
            Placeholder -> Add -> Mul -> Identity
                         /      /
            Placeholder -      c

            Subgraph_0:
                           -> Relu -
                          /         \   # noqa: W605
            Placeholder ->           Mul -> Relu -> Identity
                          \         /
                           -> Sqrt -
        """
        p = GraphDefPartitioner(_create_graph_def_front_circle())
        (
            main_graph,
            subgraphs,
            ori_input_names,
            new_input_names,
            _,
        ) = p.generate_subgraph_from_segment()
        self.assertEqual(len(subgraphs), 2)
        self.assertListEqual(ori_input_names, [["Abs:0"], ["px:0", "py:0"]])
        self.assertListEqual(
            new_input_names,
            [
                ["subgraph_0_placeholder_0"],
                ["subgraph_1_placeholder_0", "subgraph_1_placeholder_1"],
            ],
        )

        m = SimpleGraph(main_graph)
        self.assertEqual(m.num_nodes, 6)
        self.assertEqual(
            m.node(0),
            SimpleNode(
                "subgraph_0", "", ["Abs:0"], ["Relu_1"], {"subgraph_0:0": ["Relu_1"]},
            ),
        )
        self.assertEqual(
            m.node(1), SimpleNode("Relu_1", "Identity", ["subgraph_0:0"], [], {},),
        )
        self.assertEqual(
            m.node(2),
            SimpleNode(
                "subgraph_1", "", ["px:0", "py:0"], ["Abs"], {"subgraph_1:0": ["Abs"]},
            ),
        )
        self.assertEqual(
            m.node(3),
            SimpleNode(
                "px", "Placeholder", [], ["subgraph_1"], {"px:0": ["subgraph_1"]},
            ),
        )
        self.assertEqual(
            m.node(4),
            SimpleNode(
                "py", "Placeholder", [], ["subgraph_1"], {"py:0": ["subgraph_1"]},
            ),
        )
        self.assertEqual(
            m.node(5),
            SimpleNode(
                "Abs",
                "Abs",
                ["subgraph_1:0"],
                ["subgraph_0"],
                {"Abs:0": ["subgraph_0"]},
            ),
        )

        s0 = SimpleGraph(subgraphs[0])
        self.assertEqual(s0.num_nodes, 6)
        self.assertEqual(
            s0.node(0),
            SimpleNode(
                "subgraph_0_placeholder_0",
                "Placeholder",
                [],
                ["Relu", "Rsqrt"],
                {"subgraph_0_placeholder_0:0": ["Relu", "Rsqrt"]},
            ),
        )
        self.assertEqual(
            s0.node(1),
            SimpleNode(
                "Relu",
                "Relu",
                ["subgraph_0_placeholder_0:0"],
                ["mul_1"],
                {"Relu:0": ["mul_1"]},
            ),
        )
        self.assertEqual(
            s0.node(2),
            SimpleNode(
                "Rsqrt",
                "Rsqrt",
                ["subgraph_0_placeholder_0:0"],
                ["mul_1"],
                {"Rsqrt:0": ["mul_1"]},
            ),
        )
        self.assertEqual(
            s0.node(3),
            SimpleNode(
                "mul_1",
                "Mul",
                ["Relu:0", "Rsqrt:0"],
                ["Relu_1"],
                {"mul_1:0": ["Relu_1"]},
            ),
        )
        self.assertEqual(
            s0.node(4),
            SimpleNode(
                "Relu_1",
                "Relu",
                ["mul_1:0"],
                ["subgraph_0-Relu_1-0"],
                {"Relu_1:0": ["subgraph_0-Relu_1-0"]},
            ),
        )
        self.assertEqual(
            s0.node(5),
            SimpleNode("subgraph_0-Relu_1-0", "Identity", ["Relu_1:0"], [], {},),
        )

        s1 = SimpleGraph(subgraphs[1])
        self.assertEqual(s1.num_nodes, 6)
        self.assertEqual(
            s1.node(0),
            SimpleNode(
                "subgraph_1_placeholder_0",
                "Placeholder",
                [],
                ["add"],
                {"subgraph_1_placeholder_0:0": ["add"]},
            ),
        )
        self.assertEqual(
            s1.node(1),
            SimpleNode(
                "subgraph_1_placeholder_1",
                "Placeholder",
                [],
                ["add"],
                {"subgraph_1_placeholder_1:0": ["add"]},
            ),
        )
        self.assertEqual(
            s1.node(2), SimpleNode("c", "Const", [], ["mul"], {"c:0": ["mul"]},),
        )
        self.assertEqual(
            s1.node(3),
            SimpleNode(
                "add",
                "AddV2",
                ["subgraph_1_placeholder_0:0", "subgraph_1_placeholder_1:0"],
                ["mul"],
                {"add:0": ["mul"]},
            ),
        )
        self.assertEqual(
            s1.node(4),
            SimpleNode(
                "mul",
                "Mul",
                ["add:0", "c:0"],
                ["subgraph_1-mul-0"],
                {"mul:0": ["subgraph_1-mul-0"]},
            ),
        )
        self.assertEqual(
            s1.node(5), SimpleNode("subgraph_1-mul-0", "Identity", ["mul:0"], [], {},),
        )

    def test_segment_to_subgraph_front_circle_stop_at(self) -> None:
        """                           -> Relu | -
                                     /        |  \  # noqa: W605
            px -> Add -> Mul -> *ABS*         |   Mul -> Relu
                /      /             \        |  /
            py -      c               -> Sqrt | -

            Main:
                      -> Add(Identity)
                     /
            px -> Subgraph_0 --> *ABS* -> Relu
                /                     \
            py -                       -> Sqrt

            Subgraph_0:
                              -> Identity
                             /
            Placeholder -> Add -> Mul -> Identity
                         /      /
            Placeholder -      c
        """
        p = GraphDefPartitioner(
            _create_graph_def_front_circle(), outputs=["Relu", "Rsqrt", "add"]
        )
        (
            main_graph,
            subgraphs,
            ori_input_names,
            new_input_names,
            _,
        ) = p.generate_subgraph_from_segment()
        self.assertEqual(len(subgraphs), 1)
        self.assertListEqual(ori_input_names, [["px:0", "py:0"]])
        self.assertListEqual(
            new_input_names, [["subgraph_0_placeholder_0", "subgraph_0_placeholder_1"]]
        )

        m = SimpleGraph(main_graph)
        self.assertEqual(
            m.node(0),
            SimpleNode(
                "subgraph_0",
                "",
                ["px:0", "py:0"],
                ["add", "Abs"],
                {"subgraph_0:0": ["add"], "subgraph_0:1": ["Abs"]},
            ),
        )
        self.assertEqual(
            m.node(1), SimpleNode("add", "Identity", ["subgraph_0:0"], [], {},),
        )
        self.assertEqual(
            m.node(2),
            SimpleNode(
                "px", "Placeholder", [], ["subgraph_0"], {"px:0": ["subgraph_0"]},
            ),
        )
        self.assertEqual(
            m.node(3),
            SimpleNode(
                "py", "Placeholder", [], ["subgraph_0"], {"py:0": ["subgraph_0"]},
            ),
        )
        self.assertEqual(
            m.node(4),
            SimpleNode(
                "Abs",
                "Abs",
                ["subgraph_0:1"],
                ["Relu", "Rsqrt"],
                {"Abs:0": ["Relu", "Rsqrt"]},
            ),
        )
        self.assertEqual(
            m.node(5), SimpleNode("Relu", "Relu", ["Abs"], [], {},),
        )
        self.assertEqual(
            m.node(6), SimpleNode("Rsqrt", "Rsqrt", ["Abs"], [], {},),
        )

        s0 = SimpleGraph(subgraphs[0])
        self.assertEqual(s0.num_nodes, 7)
        self.assertEqual(
            s0.node(0),
            SimpleNode(
                "subgraph_0_placeholder_0",
                "Placeholder",
                [],
                ["add"],
                {"subgraph_0_placeholder_0:0": ["add"]},
            ),
        )
        self.assertEqual(
            s0.node(1),
            SimpleNode(
                "subgraph_0_placeholder_1",
                "Placeholder",
                [],
                ["add"],
                {"subgraph_0_placeholder_1:0": ["add"]},
            ),
        )
        self.assertEqual(
            s0.node(2), SimpleNode("c", "Const", [], ["mul"], {"c:0": ["mul"]},),
        )
        self.assertEqual(
            s0.node(3),
            SimpleNode(
                "add",
                "AddV2",
                ["subgraph_0_placeholder_0:0", "subgraph_0_placeholder_1:0"],
                ["mul", "subgraph_0-add-0"],
                {"add:0": ["mul", "subgraph_0-add-0"]},
            ),
        )
        self.assertEqual(
            s0.node(4),
            SimpleNode(
                "mul",
                "Mul",
                ["add:0", "c:0"],
                ["subgraph_0-mul-0"],
                {"mul:0": ["subgraph_0-mul-0"]},
            ),
        )
        self.assertEqual(
            s0.node(5), SimpleNode("subgraph_0-add-0", "Identity", ["add:0"], [], {},)
        )
        self.assertEqual(
            s0.node(6), SimpleNode("subgraph_0-mul-0", "Identity", ["mul:0"], [], {},),
        )

    def test_segment_with_control_flow(self) -> None:
        """
        After clustering, the control input on t1 should be moved onto the whole sub graph.
          Placeholder|px──────►Abs|y1────────────────────┐
                      │         .                        │
                      │         .ctl_dep                 ▼
                      │         .                      Add|out
                      │         .                        ▲
                      │         .                        │
                      │     ┌───▼────────────────────┐   │
                      │     │                        │   │
                      └─────┼─►Relu|t0──────►Relu|t1─┼───┘
                            │                        │
                            │             subgraph_0 │
                            └────────────────────────┘
        """
        graph_def = _create_graph_def_ctl_depenency()
        p = GraphDefPartitioner(graph_def, supported_list=set(["Relu"]))
        (
            main_graph,
            subgraphs,
            ori_input_names,
            new_input_names,
            _,
        ) = p.generate_subgraph_from_segment(add_function_def=True)
        self.assertEqual(len(subgraphs), 1)
        self.assertEqual(ori_input_names, [["px:0"]])
        self.assertEqual(new_input_names, [["subgraph_0_placeholder_0"]])

        subgraph_nodes = [n for n in main_graph.node if n.name.startswith("subgraph_")]
        self.assertEqual(len(subgraph_nodes), 1)
        subgraph_node_0 = subgraph_nodes[0]
        self.assertEqual(subgraph_node_0.name, "subgraph_0")
        self.assertEqual(subgraph_node_0.op, "subgraph_0")
        self.assertTrue("px" in subgraph_node_0.input)
        self.assertTrue("^y1" in subgraph_node_0.input)

        self.assertEqual(len(main_graph.library.function), 1)
        subgraph_func = main_graph.library.function[0]
        self.assertEqual(subgraph_func.signature.name, "subgraph_0")
        func_ops = [(n.name, n.op) for n in subgraph_func.node_def]
        self.assertTrue(("t0", "Relu") in func_ops)
        self.assertTrue(("t1", "Relu") in func_ops)


class SegmentToFunctionDefTest(unittest.TestCase):
    def test_segment_to_subgraph(self) -> None:
        """px -> Add --> Mul -> Relu -> *ABS* -> Add -> Sqrt
            /     \_/                       /     \ # noqa: W605
        py -                             c -       -> Relu

        Main:
        px -> Subgraph_1 -> *ABS* -> Subgraph_0 -> Sqrt(Identity)
            /                                   \   # noqa: W605
        py -                                     -> Relu

        Subgraph_1:
        Placeholder -> Add --> Mul -> Relu -> Identity
                     /     \_/
        Placeholder -

        Subgraph_0:
        Placeholder -> Add -> Sqrt -> Identity
                     /     \    # noqa: W605
                  c -       -> Identity
        """
        p = GraphDefPartitioner(_create_graph_def())

        test_data0 = np.float32(np.random.random((3, 3)))
        test_data1 = np.float32(np.random.random((3, 3)))

        # Check res for s0
        s0 = p.graph_segment_list[0]
        gd, _, _, _, _ = s0.to_graphdef()
        fd = s0.to_functiondef()
        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                tf.import_graph_def(gd, name="")
                gd_res = sess.run(
                    ["subgraph_0-Rsqrt-0:0", "subgraph_0-add_1-0:0"],
                    {"subgraph_0_placeholder_0:0": test_data0},
                )
        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                df = function._from_definition(fd)
                fd_tensors = df(
                    tf.placeholder(name="subgraph_0_placeholder_0", dtype="float32")
                )
                fd_res = sess.run(
                    fd_tensors, {"subgraph_0_placeholder_0:0": test_data0},
                )
        np.testing.assert_allclose(gd_res, fd_res)

        # Check res for s1
        s1 = p.graph_segment_list[1]
        gd, _, _, _, _ = s1.to_graphdef(1)
        fd = s1.to_functiondef(1)
        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                tf.import_graph_def(gd, name="")
                gd_res = sess.run(
                    ["subgraph_1-Relu-0:0"],
                    {
                        "subgraph_1_placeholder_0:0": test_data0,
                        "subgraph_1_placeholder_1:0": test_data1,
                    },
                )
        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                df = function._from_definition(fd)
                fd_tensors = df(
                    tf.placeholder(name="subgraph_1_placeholder_0", dtype="float32"),
                    tf.placeholder(name="subgraph_1_placeholder_1", dtype="float32"),
                )
                fd_res = sess.run(
                    [fd_tensors],
                    {
                        "subgraph_1_placeholder_0:0": test_data0,
                        "subgraph_1_placeholder_1:0": test_data1,
                    },
                )
        np.testing.assert_allclose(gd_res, fd_res)

        # Check res for replace the subgraph node as function call
        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                tf.import_graph_def(p.ori_graph._graph_def, name="")
                gd_res = sess.run(
                    ["Rsqrt:0", "Relu_1:0"], {"px:0": test_data0, "py:0": test_data1},
                )
        new_gd, _, _, _, _ = p.generate_subgraph_from_segment(True)
        # Assert the new_gd contains the updated subgraph node
        find_subgrap_node = False
        for node in new_gd.node:
            if str(node.op).startswith("subgraph_"):
                find_subgrap_node = True
        self.assertEqual(find_subgrap_node, True)
        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                tf.import_graph_def(new_gd, name="")
                new_gd_res = sess.run(
                    ["Rsqrt:0", "Relu_1:0"], {"px:0": test_data0, "py:0": test_data1},
                )
        np.testing.assert_allclose(gd_res, new_gd_res)

    def test_segment_to_subgraph_circle(self) -> None:
        """                   -> Relu --
                             /          \   # noqa: W605
            px -> Add -> Mul             Mul -> Relu
                /      /     \          /
            py -    c -       -> *ABS* -
        """
        p = GraphDefPartitioner(_create_graph_def_circle())

        test_data0 = np.float32(np.random.random((3, 3)))
        test_data1 = np.float32(np.random.random((3, 3)))

        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                tf.import_graph_def(p.ori_graph._graph_def, name="")
                gd_res = sess.run(
                    ["Relu_1:0"], {"px:0": test_data0, "py:0": test_data1},
                )
        new_gd, _, _, _, _ = p.generate_subgraph_from_segment(True)
        # Assert the new_gd contains the updated subgraph node
        find_subgrap_node = False
        for node in new_gd.node:
            if str(node.op).startswith("subgraph_"):
                find_subgrap_node = True
        self.assertEqual(find_subgrap_node, True)
        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                tf.import_graph_def(new_gd, name="")
                new_gd_res = sess.run(
                    ["Relu_1:0"], {"px:0": test_data0, "py:0": test_data1},
                )
        np.testing.assert_allclose(gd_res, new_gd_res)

    def test_segment_to_subgraph_front_circle_stop_at(self) -> None:
        """                           -> Relu | -
                                     /        |  \  # noqa: W605
            px -> Add -> Mul -> *ABS*         |   Mul -> Relu
                /      /             \        |  /
            py -      c               -> Sqrt | -

            Main:
                      -> Add(Identity)
                     /
            px -> Subgraph_0 --> *ABS* -> Relu
                /                     \
            py -                       -> Sqrt

            Subgraph_0:
                              -> Identity
                             /
            Placeholder -> Add -> Mul -> Identity
                         /      /
            Placeholder -      c
        """
        p = GraphDefPartitioner(
            _create_graph_def_front_circle(), outputs=["Relu", "Rsqrt", "add"]
        )

        test_data0 = np.float32(np.random.random((3, 3)))
        test_data1 = np.float32(np.random.random((3, 3)))

        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                tf.import_graph_def(p.ori_graph._graph_def, name="")
                gd_res = sess.run(
                    ["Relu:0", "Rsqrt:0", "add:0"],
                    {"px:0": test_data0, "py:0": test_data1},
                )
        new_gd, _, _, _, _ = p.generate_subgraph_from_segment(True)
        # Assert the new_gd contains the updated subgraph node
        find_subgrap_node = False
        for node in new_gd.node:
            if str(node.op).startswith("subgraph_"):
                find_subgrap_node = True
        self.assertEqual(find_subgrap_node, True)
        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                tf.import_graph_def(new_gd, name="")
                new_gd_res = sess.run(
                    ["Relu:0", "Rsqrt:0", "add:0"],
                    {"px:0": test_data0, "py:0": test_data1},
                )
        np.testing.assert_allclose(gd_res, new_gd_res)

    def test_segment_to_subgraph_while_loop(self) -> None:
        """              ------------------
           px -> Mul -> | -> Add -> Mul -> | -> Relu -> Sqrt
               /        |  /      /        |
           py -    c -> | --------         |
                        |                  |
                        | i -> Add -> i    |
                        |    /             |
                        | 1 -              |
                         ------------------

            Main:
                          ------------------
            px -> Mul -> | -> Subgraph_1 -> | -> Subgraph_0 -> Sqrt(Identity)
                /        |  /               |
            py -    c -> | -                |
                         |                  |
                         | i -> Add -> i    |
                         |    /             |
                         | 1 -              |
                          ------------------

            Subgraph_0:
            Placeholder -> Add -> Mul -> Identity
                         /      /
            Placeholder --------

            Subgraph_1:
            Placeholder -> Relu -> Sqrt -> Identity
        """
        p = GraphDefPartitioner(_create_graph_def_while_loop())

        self.assertEqual(len(p.graph_segment_list), 2)
        self.assertListEqual(
            p.graph_segment_list[0].node_names,
            ["while/mul", "while/Identity_2", "while/add_1"],
        )
        self.assertListEqual(p.graph_segment_list[1].node_names, ["Rsqrt", "Relu"])

        (
            main_graph,
            subgraphs,
            ori_input_names,
            new_input_names,
            _,
        ) = p.generate_subgraph_from_segment(True)
        self.assertListEqual(
            ori_input_names,
            [["while/add_1/Enter:0", "while/Switch_2:1"], ["while/Exit_2:0"]],
        )
        self.assertListEqual(
            new_input_names,
            [
                ["subgraph_0_placeholder_0", "subgraph_0_placeholder_1"],
                ["subgraph_1_placeholder_0"],
            ],
        )

        s0 = SimpleGraph(subgraphs[0])
        self.assertEqual(s0.num_nodes, 6)
        self.assertEqual(
            s0.node(0),
            SimpleNode(
                "subgraph_0_placeholder_0",
                "Placeholder",
                [],
                ["while/add_1", "while/mul"],
                {"subgraph_0_placeholder_0:0": ["while/add_1", "while/mul"]},
            ),
        )
        self.assertEqual(
            s0.node(1),
            SimpleNode(
                "subgraph_0_placeholder_1",
                "Placeholder",
                [],
                ["while/Identity_2"],
                {"subgraph_0_placeholder_1:0": ["while/Identity_2"]},
            ),
        )
        self.assertEqual(
            s0.node(2),
            SimpleNode(
                "while/Identity_2",
                "Identity",
                ["subgraph_0_placeholder_1:0"],
                ["while/add_1"],
                {"while/Identity_2:0": ["while/add_1"]},
            ),
        )
        self.assertEqual(
            s0.node(3),
            SimpleNode(
                "while/add_1",
                "AddV2",
                ["while/Identity_2:0", "subgraph_0_placeholder_0:0"],
                ["while/mul"],
                {"while/add_1:0": ["while/mul"]},
            ),
        )
        self.assertEqual(
            s0.node(4),
            SimpleNode(
                "while/mul",
                "Mul",
                ["while/add_1:0", "subgraph_0_placeholder_0:0"],
                ["subgraph_0-while/mul-0"],
                {"while/mul:0": ["subgraph_0-while/mul-0"]},
            ),
        )
        self.assertEqual(
            s0.node(5),
            SimpleNode("subgraph_0-while/mul-0", "Identity", ["while/mul:0"], [], {}),
        )

        s1 = SimpleGraph(subgraphs[1])
        self.assertEqual(s1.num_nodes, 4)
        self.assertEqual(
            s1.node(0),
            SimpleNode(
                "subgraph_1_placeholder_0",
                "Placeholder",
                [],
                ["Relu"],
                {"subgraph_1_placeholder_0:0": ["Relu"]},
            ),
        )
        self.assertEqual(
            s1.node(1),
            SimpleNode(
                "Relu",
                "Relu",
                ["subgraph_1_placeholder_0:0"],
                ["Rsqrt"],
                {"Relu:0": ["Rsqrt"]},
            ),
        )
        self.assertEqual(
            s1.node(2),
            SimpleNode(
                "Rsqrt",
                "Rsqrt",
                ["Relu:0"],
                ["subgraph_1-Rsqrt-0"],
                {"Rsqrt:0": ["subgraph_1-Rsqrt-0"]},
            ),
        )
        self.assertEqual(
            s1.node(3),
            SimpleNode("subgraph_1-Rsqrt-0", "Identity", ["Rsqrt:0"], [], {}),
        )

        test_data0 = np.float32(np.random.random((3, 3)))
        test_data1 = np.float32(np.random.random((3, 3)))
        test_n = np.int32(10)

        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                tf.import_graph_def(p.ori_graph._graph_def, name="")
                gd_res = sess.run(
                    ["Rsqrt:0"],
                    {"px:0": test_data0, "py:0": test_data1, "n:0": test_n},
                )
        new_gd, _, _, _, _ = p.generate_subgraph_from_segment(True)
        # Assert the new_gd contains the updated subgraph node
        find_subgrap_node = False
        for node in new_gd.node:
            if str(node.op).startswith("subgraph_"):
                find_subgrap_node = True
        self.assertEqual(find_subgrap_node, True)
        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                tf.import_graph_def(new_gd, name="")
                new_gd_res = sess.run(
                    ["Rsqrt:0"],
                    {"px:0": test_data0, "py:0": test_data1, "n:0": test_n},
                )
        np.testing.assert_allclose(gd_res, new_gd_res)

    def test_quirky_attr(self) -> None:
        g = SimpleGraph(_create_graph_def_with_quirky_type_attr())
        self.assertEqual(g.num_nodes, 11)
        self.assertListEqual(g.output_nodes(), ["Rsqrt", "Cast"])

        expected_inputs: List[List[int]] = [
            [],  # node 0
            [],  # node 1
            [],  # node 2
            [0, 1],  # node 3
            [3, 3],  # node 4
            [4],  # node 5
            [5],  # node 6
            [6, 2],  # node 7
            [7],  # node 8
            [7],  # node 9
            [9],  # node 10
        ]

        for i in range(g.num_nodes):
            self.assertListEqual(g.input_nodes_index(i), expected_inputs[i])

        # Check nodes
        self.assertEqual(
            g.node(0), SimpleNode("px", "Placeholder", [], ["add"], {"px:0": ["add"]})
        )
        self.assertEqual(
            g.node(1), SimpleNode("py", "Placeholder", [], ["add"], {"py:0": ["add"]})
        )
        self.assertEqual(
            g.node(2), SimpleNode("c", "Const", [], ["add_1"], {"c:0": ["add_1"]})
        )
        self.assertEqual(
            g.node(3),
            SimpleNode(
                "add",
                "AddV2",
                ["px:0", "py:0"],
                ["mul", "mul"],
                {"add:0": ["mul", "mul"]},
            ),
        )
        self.assertEqual(
            g.node(4),
            SimpleNode("mul", "Mul", ["add:0", "add:0"], ["Relu"], {"mul:0": ["Relu"]}),
        )
        self.assertEqual(
            g.node(5),
            SimpleNode("Relu", "Relu", ["mul:0"], ["Abs"], {"Relu:0": ["Abs"]}),
        )
        self.assertEqual(
            g.node(6),
            SimpleNode("Abs", "Abs", ["Relu:0"], ["add_1"], {"Abs:0": ["add_1"]}),
        )
        self.assertEqual(
            g.node(7),
            SimpleNode(
                "add_1",
                "AddV2",
                ["Abs:0", "c:0"],
                ["Rsqrt", "Relu_1"],
                {"add_1:0": ["Rsqrt", "Relu_1"]},
            ),
        )
        self.assertEqual(g.node(8), SimpleNode("Rsqrt", "Rsqrt", ["add_1:0"], [], {}))
        self.assertEqual(
            g.node(9),
            SimpleNode("Relu_1", "Relu", ["add_1:0"], ["Cast"], {"Relu_1:0": ["Cast"]}),
        )
        self.assertEqual(g.node(10), SimpleNode("Cast", "Cast", ["Relu_1:0"], [], {}))


if __name__ == "__main__":
    unittest.main()

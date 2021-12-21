import unittest
import torch
import torch_addons.utils as utils
from torch_addons.testing.common_utils import Feedforward, TestCase


class TestUtils(TestCase):

    def setUp(self):
        ff_net = Feedforward(64, 10)
        self.ff_net = torch.jit.script(ff_net)

    def test_graph_nodes(self):
        graph_node_list = self.ff_net.graph.node_list()
        graph_nodes = [n for n in self.ff_net.graph.nodes()]
        self.assertEqual(graph_node_list, graph_nodes)

    def test_graph_inputs(self):
        graph_input_list = self.ff_net.graph.input_list()
        graph_inputs = [n for n in self.ff_net.graph.inputs()]
        self.assertEqual(graph_input_list, graph_inputs)

    def test_graph_outputs(self):
        graph_output_list = self.ff_net.graph.output_list()
        graph_outputs = [n for n in self.ff_net.graph.outputs()]
        self.assertEqual(graph_output_list, graph_outputs)

    def test_graph_topology(self):
        graph = self.ff_net.graph.copy()
        self.assertTrue(utils.graph_in_topology_order(graph))

        nodes = graph.node_list()
        self.assertTrue(len(nodes) > 2)
        nodes[-1].moveBefore(nodes[0])
        self.assertTrue(graph.node_list()[0] == nodes[-1])
        self.assertFalse(utils.graph_in_topology_order(graph))

    def test_control_dependecies(self):
        def loop_fun(init):
            vals = [1, 2, 3]
            niter = 3
            mid = 2
            for i in range(niter):
                if i < mid:
                    init = init + vals[i] + niter
                else:
                    init = init + vals[i]
            return init
        loop_fun = torch.jit.script(loop_fun)
        graph = loop_fun.graph
        loop = [n for n in graph.nodes() if n.kind() == 'prim::Loop'][0]
        deps = loop.control_deps()
        loop_inputs = loop.input_list()
        self.assertTrue(all(d not in loop_inputs for d in deps))

        vals_in_toplevel = graph.input_list()
        for n in graph.nodes():
            vals_in_toplevel += n.output_list()
        self.assertTrue(all(d in vals_in_toplevel for d in deps))


if __name__ == '__main__':
    unittest.main()

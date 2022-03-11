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
import torch
from torch_blade import tensorrt
from torch_blade import tools
from torch_blade.clustering import support_fusion_algorithm as algorithm
from torch_blade.clustering import support_fusion_group
from torch_blade.testing.common_utils import Feedforward, TestCase
from tests.tensorrt import skipIfNoTensorRT


@skipIfNoTensorRT()
class TestTensorRTSupportFusion(TestCase):
    def create_ffg(self):
        input = torch.ones([10, 10]).cuda()
        net = Feedforward(10, 10)
        net.eval().cuda()
        module = torch.jit.trace(net, input)
        module = tools.freeze_module(module._c, disableShapePeephole=False)
        return module.forward.graph

    def test_support_fusion_algorithm(self):
        graph = self.create_ffg()

        unsupported = tensorrt.get_unsupported_nodes(graph)
        self.assertEqual(len(unsupported), 0)
        fusion_groups = algorithm.group_supported_clusters(graph, unsupported)
        self.assertEqual(len(fusion_groups), 1)

        non_const_nodes = [n for n in graph.node_list() if n.kind() != "prim::Constant"]
        self.assertEqual(len(fusion_groups[0]), len(non_const_nodes))

    def test_support_fusion_group(self):
        graph = self.create_ffg()

        unsupported = tensorrt.get_unsupported_nodes(graph)
        support_fusion_group.supported_node_fusion(graph, graph, unsupported)
        self.assertEqual(len(graph.node_list()), 1)
        self.assertEqual(graph.node_list()[0].kind(), "prim::FusionGroup")


if __name__ == "__main__":
    unittest.main()

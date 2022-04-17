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

import collections
import unittest
import torch
from torch.nn import functional as F
from torch_blade import tensorrt
from torch_blade import utils
from torch_blade import tools
from torch_blade import Config
from torch_blade.logging import logger
from torch_blade.testing.common_utils import Feedforward, TestCase
from tests.tensorrt import skipIfNoTensorRT
from torch_blade.onnx_backends.backend_testbed import OnnxBackendChecker


@skipIfNoTensorRT()
class TestTensorRTSupportInfo(TestCase):
    def test_support_info(self):
        input = torch.ones([10, 10]).cuda()
        net = Feedforward(10, 10)
        net.eval().cuda()
        module = torch.jit.trace(net, input)
        module = tools.freeze_module(module._c, disableShapePeephole=False)
        graph = module.forward.graph

        unsupported = tensorrt.get_unsupported_nodes(graph)
        self.assertEqual(len(unsupported), 0)

    def test_empty_onnx_export(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.linear = torch.nn.Linear(3, 4)
                self.dropout = torch.nn.Dropout(p=0.8)

            def forward(self, x):
                x = self.linear(x)
                x = self.dropout(x)
                return x.contiguous().detach()

        model = Model().cuda().eval()
        module = torch.jit.trace(model, torch.ones([2, 3]).cuda())
        module = tools.freeze_module(module._c, disableShapePeephole=False)
        graph = module.forward.graph

        unsupported = tensorrt.get_unsupported_nodes(graph)
        self.assertEqual(len(unsupported), 0)

    def test_inplace_safety(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super(BasicBlock, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=3, padding=1)
                self.conv2 = torch.nn.Conv2d(10, 3, kernel_size=3, padding=1)
                self.conv3 = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)
                self.bnorm = torch.nn.BatchNorm2d(3)

            def forward_inplace(self, x):
                out = self.conv1(x)
                # this inplace bias is supported
                out += 1
                # this inplace relu_ is supported
                out = F.relu_(out)
                out = self.conv2(out)
                # this inplace relu_ is supported
                out = F.relu_(out)
                shortcut = out
                # this inplace add_ is supported
                out += shortcut
                shortcut = out
                out = self.conv3(out)
                out = self.bnorm(out)
                # this inplace add_ is supported
                out += shortcut
                out1 = out[:, :1, :, :]
                out2 = out[:, 1:, :, :]
                out1 = F.relu_(out1)
                out2 = F.relu_(out2)
                out[:, :1, :, :] = out1
                out[:, 1:, :, :] = out2
                return out

            def forward_no_inplace(self, x):
                out = self.conv1(x)
                out = out + 1
                out = F.relu(out)
                out = self.conv2(out)
                out = F.relu(out)
                shortcut = out
                out = out + shortcut
                shortcut = out
                out = self.conv3(out)
                out = self.bnorm(out)
                out = out + shortcut
                out = F.relu(out)
                return out

        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.block1 = BasicBlock()
                self.block2 = BasicBlock()

            def forward(self, x):
                out1 = self.block1.forward_inplace(x)
                out1 = self.block2.forward_inplace(out1)
                out2 = self.block1.forward_no_inplace(x)
                out2 = self.block2.forward_no_inplace(out2)
                return out1, out2

        model = Model()
        model.eval()
        model.cuda()

        batch = torch.ones([1, 3, 224, 224])
        batch = batch.cuda()
        out1, out2 = model(batch)
        self.assertEqual(out1, out2)
        traced_model = torch.jit.trace(model, batch)
        frozen_module = tools.freeze_module(traced_model._c, disableShapePeephole=False)
        graph = frozen_module.forward.graph
        ops_counter = utils.list_ops_count(graph)
        unspt_counter = collections.Counter()
        unsupported = tensorrt.get_unsupported_nodes(graph)
        for node in unsupported:
            unspt_counter[node.kind()] += 1
        self.assertEqual(ops_counter["aten::slice"], unspt_counter["aten::slice"])
        self.assertEqual(ops_counter["aten::view"], unspt_counter["aten::view"])
        self.assertEqual(ops_counter["aten::copy_"], unspt_counter["aten::copy_"])
        self.assertEqual(ops_counter["aten::expand"], unspt_counter["aten::expand"])
        self.assertEqual(unspt_counter["aten::relu_"], 4)
        logger.info(ops_counter)
        logger.info(unspt_counter)
        self.assertEqual(unspt_counter["aten::add_"], 0)

    def test_inplace_safety_another(self):
        def op(x):
            return x + 1

        def op_(x):
            x -= 1
            return x

        def _count_unsupported(unspt):
            unspt_counter = collections.Counter()
            for node in unspt:
                unspt_counter[node.kind()] += 1
            return unspt_counter

        def _count_graph(graph):
            unsupported = tensorrt.get_unsupported_nodes(graph, ignore_device=True)
            return _count_unsupported(unsupported)

        def _count_model(model):
            model.eval().cuda()
            input = torch.zeros([4]).cuda()
            output = model(input)
            traced_module = torch.jit.trace(model, (input,))
            graph = traced_module.graph
            return _count_graph(graph)

        class Model1(torch.nn.Module):
            """
            Within this model, torch.jit.trace will produce graph like:
                %2 : Float = aten::add(%1, some_constant)
                %3 : Float = aten::sub_(%2, some_constant)
                %4 : Float = aten::add(%3, some_constant)

            The input of the third node is %3 instead of %2 which is not consistent with the definition of the
            corresponding nn.Module. So the inplace node aten::sub_ is the last consumer of its inputs which make it
            inplace-safe, and therefore all the nodes in this graph is inplace-safe.

            The same phenomenon occurs in model2. So we manually add two graphs that have 'correct' topology structures
            with corresponding nn.Module (i.e. Model1 and Model2) and use them as UTs.
            """

            def forward(self, x):
                x1 = op(x)
                x2 = op_(x1)
                x3 = op(x1)
                return x3

        class Model2(torch.nn.Module):
            def forward(self, x):
                x1 = op(x)
                x2 = op_(x1)  # support
                x3 = op_(x2)  # support
                x4 = op(x3)
                x5 = op_(x3)  # not support
                x6 = op_(x5)  # not support
                x7 = op(x3)
                return x7

        unspt_counter = _count_model(Model1())
        self.assertEqual(unspt_counter["aten::sub_"], 0)
        unspt_counter = _count_model(Model2())
        self.assertEqual(unspt_counter["aten::sub_"], 0)

        if utils.torch_version_number() >= utils.parse_version("1.8.1"):
            graph1 = torch.parse_ir(
                """
                    graph( %x.1 : Float(4)):
                      %1 : int = prim::Constant[value=1]()
                      %2 : Float(4) = aten::add(%x.1, %1, %1)
                      %3 : int = prim::Constant[value=1]()
                      %4 : Float(4) = aten::sub_(%2, %3, %3)
                      %5 : int = prim::Constant[value=1]()
                      %6 : Float(4) = aten::add(%2, %5, %5)
                      return (%6)
                """
            )

            graph2 = torch.parse_ir(
                """
                    graph( %x.1 : Float(4)):
                      %1 : int = prim::Constant[value=1]()
                      %2 : Float(4) = aten::add(%x.1, %1, %1)
                      %3 : int = prim::Constant[value=1]()
                      %4 : Float(4) = aten::sub_(%2, %3, %3)
                      %5 : int = prim::Constant[value=1]()
                      %6 : Float(4) = aten::sub_(%4, %5, %5)
                      %7 : int = prim::Constant[value=1]()
                      %8 : Float(4) = aten::add(%6, %7, %7)
                      %9 : int = prim::Constant[value=1]()
                      %10 : Float(4) = aten::sub_(%6, %9, %9)
                      %11 : int = prim::Constant[value=1]()
                      %12 : Float(4) = aten::sub_(%10, %11, %11)
                      %13 : int = prim::Constant[value=1]()
                      %14 : Float(4) = aten::add(%6, %13, %13)
                      return (%14)
                """
            )
        else:
            graph1 = torch.parse_ir(
                """
                    graph( %x.1 : Float(4:1)):
                      %1 : int = prim::Constant[value=1]()
                      %2 : Float(4:1) = aten::add(%x.1, %1, %1)
                      %3 : int = prim::Constant[value=1]()
                      %4 : Float(4:1) = aten::sub_(%2, %3, %3)
                      %5 : int = prim::Constant[value=1]()
                      %6 : Float(4:1) = aten::add(%2, %5, %5)
                      return (%6)
                """
            )

            graph2 = torch.parse_ir(
                """
                    graph( %x.1 : Float(4:1)):
                      %1 : int = prim::Constant[value=1]()
                      %2 : Float(4:1) = aten::add(%x.1, %1, %1)
                      %3 : int = prim::Constant[value=1]()
                      %4 : Float(4:1) = aten::sub_(%2, %3, %3)
                      %5 : int = prim::Constant[value=1]()
                      %6 : Float(4:1) = aten::sub_(%4, %5, %5)
                      %7 : int = prim::Constant[value=1]()
                      %8 : Float(4:1) = aten::add(%6, %7, %7)
                      %9 : int = prim::Constant[value=1]()
                      %10 : Float(4:1) = aten::sub_(%6, %9, %9)
                      %11 : int = prim::Constant[value=1]()
                      %12 : Float(4:1) = aten::sub_(%10, %11, %11)
                      %13 : int = prim::Constant[value=1]()
                      %14 : Float(4:1) = aten::add(%6, %13, %13)
                      return (%14)
                """
            )
        unspt_counter = _count_graph(graph1)
        self.assertEqual(unspt_counter["aten::sub_"], 1)
        unspt_counter = _count_graph(graph2)
        self.assertEqual(unspt_counter["aten::sub_"], 2)

    def test_graph_input_inplace_safe(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return F.relu_(x)

        batch = torch.Tensor([1, -1, 1, -1])
        batch = batch.cuda()
        model = Model().eval().cuda()
        traced_model = torch.jit.trace(model, batch)
        self.assertEqual(batch, torch.Tensor([1, 0, 1, 0]))

        frozen_module = torch._C._freeze_module(traced_model._c)
        graph = frozen_module.forward.graph
        unspt_counter = collections.Counter()
        unsupported = tensorrt.get_unsupported_nodes(graph)
        for node in unsupported:
            unspt_counter[node.kind()] += 1
        self.assertEqual(unspt_counter["aten::relu_"], 1)

    def test_view_kinds_0(self):
        if utils.torch_version_number() >= utils.parse_version("1.8.1"):
            graph = torch.parse_ir(
                """
                    graph( %x.1 : Float(1, 1, 1)):
                      %1 : int = prim::Constant[value=0]()
                      %2 : int = prim::Constant[value=1]()
                      %3 : Float(1, 1) = aten::select(%x.1, %1, %2)
                      %4 : int = prim::Constant[value=0]()
                      %5 : int = prim::Constant[value=1]()
                      %6 : Float(1) = aten::select(%3, %4, %5)
                      %7 : int = prim::Constant[value=1]()
                      %8 : int = prim::Constant[value=1]()
                      %9 : Float(1) = aten::add(%6, %7, %8)
                      return (%9)
                """
            )
        else:
            graph = torch.parse_ir(
                """
                    graph( %x.1 : Float(1:1, 1:1, 1:1)):
                      %1 : int = prim::Constant[value=0]()
                      %2 : int = prim::Constant[value=1]()
                      %3 : Float(1:1, 1:1) = aten::select(%x.1, %1, %2)
                      %4 : int = prim::Constant[value=0]()
                      %5 : int = prim::Constant[value=1]()
                      %6 : Float(1:1) = aten::select(%3, %4, %5)
                      %7 : int = prim::Constant[value=1]()
                      %8 : int = prim::Constant[value=1]()
                      %9 : Float(1:1) = aten::add(%6, %7, %8)
                      return (%9)
                """
            )
        unsupported = tensorrt.get_unsupported_nodes(graph, True)
        self.assertEqual(len(unsupported), 0)

    def test_view_kinds_1(self):
        if utils.torch_version_number() >= utils.parse_version("1.8.1"):
            graph = torch.parse_ir(
                """
                    graph( %x.1 : Float(1, 1, 1)):
                      %1 : int = prim::Constant[value=0]()
                      %2 : int = prim::Constant[value=1]()
                      %3 : Float(1, 1) = aten::select(%x.1, %1, %2)
                      %4 : int = prim::Constant[value=0]()
                      %5 : int = prim::Constant[value=1]()
                      %6 : Float(1) = aten::select(%3, %4, %5)
                      %7 : int = prim::Constant[value=1]()
                      %8 : int = prim::Constant[value=1]()
                      %9 : Float(1) = aten::add_(%6, %7, %8)
                      return (%9)
                """
            )
        else:
            graph = torch.parse_ir(
                """
                    graph( %x.1 : Float(1:1, 1:1, 1:1)):
                      %1 : int = prim::Constant[value=0]()
                      %2 : int = prim::Constant[value=1]()
                      %3 : Float(1:1, 1:1) = aten::select(%x.1, %1, %2)
                      %4 : int = prim::Constant[value=0]()
                      %5 : int = prim::Constant[value=1]()
                      %6 : Float(1:1) = aten::select(%3, %4, %5)
                      %7 : int = prim::Constant[value=1]()
                      %8 : int = prim::Constant[value=1]()
                      %9 : Float(1:1) = aten::add_(%6, %7, %8)
                      return (%9)
                """
            )
        unsupported = tensorrt.get_unsupported_nodes(graph, True)
        self.assertEqual(len(unsupported), 3)

    def test_view_kinds_2(self):
        if utils.torch_version_number() >= utils.parse_version("1.8.1"):
            graph = torch.parse_ir(
                """
                    graph( %x.1 : Float(1, 1, 1)):
                      %1 : int = prim::Constant[value=0]()
                      %2 : int = prim::Constant[value=1]()
                      %3 : Float(1, 1) = aten::select(%x.1, %1, %2)
                      %4 : int = prim::Constant[value=0]()
                      %5 : int = prim::Constant[value=1]()
                      %6 : Float(1, 1) = aten::add_(%3, %4, %5)
                      %7 : int = prim::Constant[value=1]()
                      %8 : int = prim::Constant[value=1]()
                      %9 : Float(1) = aten::select(%3, %7, %8)
                      return (%9)
                """
            )
        else:
            graph = torch.parse_ir(
                """
                    graph( %x.1 : Float(1:1, 1:1, 1:1)):
                      %1 : int = prim::Constant[value=0]()
                      %2 : int = prim::Constant[value=1]()
                      %3 : Float(1:1, 1:1) = aten::select(%x.1, %1, %2)
                      %4 : int = prim::Constant[value=0]()
                      %5 : int = prim::Constant[value=1]()
                      %6 : Float(1:1, 1:1) = aten::add_(%3, %4, %5)
                      %7 : int = prim::Constant[value=1]()
                      %8 : int = prim::Constant[value=1]()
                      %9 : Float(1:1) = aten::select(%3, %7, %8)
                      return (%9)
                """
            )
        unsupported = tensorrt.get_unsupported_nodes(graph, True)
        self.assertEqual(len(unsupported), 3)

        # NOTE: this unsupported set length should be 3 (two aten::select and one aten::add_)
        # However, due to a flaw of the inplace safety check algorithm, aten::add_ is excluded
        # in the set.
        # todo: fix this error.
        # graph = torch.parse_ir(
        #     '''
        #         graph( %x.1 : Float(1:1, 1:1, 1:1)):
        #           %1 : int = prim::Constant[value=0]()
        #           %2 : int = prim::Constant[value=1]()
        #           %3 : Float(1:1, 1:1, 1:1) = aten::add(%x.1, %1, %2)
        #           %4 : int = prim::Constant[value=0]()
        #           %5 : int = prim::Constant[value=1]()
        #           %6 : Float(1:1, 1:1) = aten::select(%3, %4, %5)
        #           %7 : Float(1:1, 1:1) = aten::add_(%3, %4, %5)
        #           %8 : int = prim::Constant[value=1]()
        #           %9 : int = prim::Constant[value=1]()
        #           %10 : Float(1:1) = aten::select(%6, %8, %9)
        #           return (%9)
        #     '''
        # )
        # unsupported = tensorrt.get_unsupported_nodes(graph, True)
        # self.assertEqual(len(unsupported), 2)

@skipIfNoTensorRT()
class TestManRules(TestCase):
    def _make_check(self, graph, target):
        checker = OnnxBackendChecker(graph, tensorrt.is_onnx2trt_supported, "TensorRT")
        is_supported = checker()
        self.assertEqual(is_supported, target)

    def test_aten_mul(self):
        graph = torch.parse_ir(
            """
                graph(%0 : int[]):
                  %1 : int = prim::Constant[value=1]()
                  %3 : int = aten::mul(%0, %1)
                  return (%3)
            """
        )
        self._make_check(graph, False)

    def test_aten_add(self):
        graph = torch.parse_ir(
            """
                graph(%0 : int[], %1 : int[]):
                  %2 : int[] = aten::add(%0, %1)
                  return (%2)
            """
        )
        self._make_check(graph, False)

    def test_aten_eq(self):
        graph = torch.parse_ir(
            """
                graph(%0 : int[]):
                  %1 : int = prim::Constant[value=1]()
                  %2 : int[] = prim::ListConstruct(%1)
                  %3 : bool = aten::eq(%0, %2)
                  return (%3)
            """
        )
        self._make_check(graph, False)

    def test_const_fold_before_export(self):
        graph = torch.parse_ir(
            """
            graph(%input0.2 : Float(1:165888, 512:324, 18:18, 18:1, requires_grad=0, device=cuda:0)):
                %1 : None = prim::Constant() # :0:0
                %2 : bool = prim::Constant[value=1]()
                %3 : float[] = prim::Constant[value=[2., 2.]]()
                %x1.3 : Float(1:663552, 512:1296, 36:36, 36:1, requires_grad=0, device=cuda:0) = aten::upsample_bilinear2d(%input0.2, %1, %2, %3)
                return (%x1.3)
            """
        )
        cfg = Config.get_current_context_or_new().clone()
        cfg.customize_onnx_opset_version = 11
        with cfg:
            self._make_check(graph, True)

    def test_scalar_input_on_graph(self):
        graph = torch.parse_ir(
            """
            graph(%x.3 : Float(1:64, 64:1, 1:1, 1:1, requires_grad=0, device=cuda:0),
                    %1 : int):
                %2 : int = prim::Constant[value=-1]()
                %3 : int[] = prim::ListConstruct(%1, %2)
                %input.14 : Float(1:64, 64:1, requires_grad=0, device=cuda:0) = aten::view(%x.3, %3)
                return (%input.14)
            """
        )
        self._make_check(graph, True)


if __name__ == "__main__":
    unittest.main()

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
import io
import onnx
import torch
import torch_blade
import torch_blade.pass_manager as pass_manager
from torch_blade import tools
from torch_blade.testing.common_utils import Feedforward, TestCase
from tests.tensorrt import skipIfNoTensorRT


@skipIfNoTensorRT()
class TestTrtOnnxParser(TestCase):
    def test_onnx_parser(self):
        input = torch.ones([10, 10])
        net = Feedforward(10, 10)
        net.eval()
        module = torch.jit.trace(net, input)
        is_static = False
        module = tools.freeze_module(module._c, disableShapePeephole=is_static)
        graph = module.forward.graph
        graph, _ = pass_manager._jit_pass_lower_to_onnx(graph)
        graph.eraseInput(0)
        proto = pass_manager._export_onnx(graph, dict())
        is_support = torch_blade.tensorrt.is_onnx2trt_supported(proto)
        self.assertTrue(is_support)
        onnx_model = onnx.load_from_string(proto)
        self.assertGreaterEqual(onnx.IR_VERSION, onnx_model.ir_version)
        onnx.checker.check_model(onnx_model)

    def test_feedforwad_export(self):
        input = torch.ones([10, 10])
        net = Feedforward(10, 10)
        net.eval()

        f = io.BytesIO()
        module = torch.onnx.export(net, input, f)
        onnx_model = onnx.load_from_string(f.getvalue())
        self.assertGreaterEqual(onnx.IR_VERSION, onnx_model.ir_version)
        onnx.checker.check_model(onnx_model)


if __name__ == '__main__':
    unittest.main()

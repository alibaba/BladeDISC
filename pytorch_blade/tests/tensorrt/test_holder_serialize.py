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
import os
import tempfile
import unittest

import onnx
import torch
from tests.tensorrt import skipIfNoTensorRT
from torch import nn
from torch.nn import functional as F
from torch_blade import tensorrt, tools
from torch_blade.clustering.support_fusion_group import supported_node_fusion
from torch_blade.config import Config
from torch_blade.optimization import optimize
from torch_blade.testing.common_utils import Feedforward, TestCase


@skipIfNoTensorRT()
class TestHolderSerialize(TestCase):
    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_holder_serialize(self):
        input = torch.ones([10, 10]).cuda()
        net = Feedforward(10, 10)
        net.cuda().eval()
        module = torch.jit.trace(net, input)
        module = tools.freeze_module(module._c, disableShapePeephole=False)
        graph = module.forward.graph
        unspt_nodes = tensorrt.get_unsupported_nodes(graph, dict())
        supported_node_fusion(graph, graph, unspt_nodes)
        fusion_group = [n for n in graph.nodes() if n.kind() == "prim::FusionGroup"]
        self.assertEqual(len(fusion_group), 1)
        tensorrt.trt_engine_conversion(module)
        engines = tensorrt.collect_engines(module)
        self.assertGreaterEqual(len(engines), 1)

        for _name, trt_engine in engines:
            saved_onnx_model = tempfile.NamedTemporaryFile()
            trt_engine.dump_model_proto(saved_onnx_model.name)
            onnx_model = onnx.load(saved_onnx_model.name)
            self.assertGreaterEqual(onnx.IR_VERSION, onnx_model.ir_version)
            onnx.checker.check_model(onnx_model)

            attr_keys = set(trt_engine.get_attr_keys())
            self.assertEqual(
                set({"module_graph", "fallback_module", "attr_debug_name"}),
                attr_keys,
            )

    def _test_dynamic_shape_engines(self, model, shapes):
        def check_output(model, opt_model, shape):
            inp = torch.randn(*shape).cuda()
            origin_out = model(inp)
            new_out = opt_model(inp)
            self.assertEqual(new_out, origin_out)

        test_shapes = [
            [1, 1, 5, 5],
            [1, 1, 6, 6],
            [1, 1, 8, 8],
            [1, 1, 10, 10],
        ]
        new_cfg = Config.get_current_context_or_new()
        new_cfg.dynamic_tuning_shapes = shapes
        new_cfg.optimization_pipeline = tensorrt.backend_name()
        with new_cfg:
            model = model.eval().cuda()
            inp = torch.randn(1, 1, 5, 5).cuda()
            opt_model = optimize(model, allow_tracing=True, model_inputs=inp)
            for shape in test_shapes:
                check_output(model, opt_model, shape)

            saved_onnx_model = tempfile.NamedTemporaryFile()
            opt_model.trt_grp0_len24_0.dump_model_proto(
                saved_onnx_model.name
            )
            onnx_model = onnx.load(saved_onnx_model.name)
            self.assertGreaterEqual(onnx.IR_VERSION, onnx_model.ir_version)
            onnx.checker.check_model(onnx_model)

    @unittest.skipIf(
        torch.cuda.is_available() and torch.cuda.get_device_capability() <= (6, 1),
        "TensorRT precision error",
    )
    def test_dynamic_shape_engines(self):
        class MyModule1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(1, 2, 3, padding=1)
                self.conv2 = torch.nn.Conv2d(2, 3, 3, padding=1)
                self.conv3 = torch.nn.Conv2d(3, 4, 3, padding=1)

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                return x

        class MyModule2(MyModule1):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = super().forward(x)
                return x.flatten(2, 3)

        shapes = {
            "min": [
                [1, 1, 5, 5],
            ],
            "max": [
                [1, 1, 10, 10],
            ],
            "opts": [
                [
                    [1, 1, 6, 6],
                ],
            ],
        }
        self._test_dynamic_shape_engines(MyModule1(), shapes)
        new_cfg = Config.get_current_context_or_new()

        # Ref: https://github.com/pytorch/pytorch/blob/master/torch/onnx/symbolic_opset9.py#L2485
        #     'aten::flatten' would be convert to Constant onnx::Reshape.
        # We would like it's left over during clustering onnx subgraph
        new_cfg.customize_op_black_list = ["aten::flatten"]
        with new_cfg:
            self._test_dynamic_shape_engines(MyModule2(), shapes)

        # multiple dynamic ranges
        shapes1 = {
            "min": [
                [1, 1, 5, 5],
            ],
            "max": [
                [1, 1, 7, 7],
            ],
            "opts": [
                [
                    [1, 1, 6, 6],
                ],
            ],
        }

        shapes2 = {
            "min": [
                [1, 1, 7, 7],
            ],
            "max": [
                [1, 1, 10, 10],
            ],
            "opts": [
                [
                    [1, 1, 8, 8],
                ],
            ],
        }
        shapes = [shapes1, shapes2]
        self._test_dynamic_shape_engines(MyModule1(), shapes)

    def test_trt_ptq_engine(self):
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 256, (3, 3), padding=1, bias=False)
                self.conv2 = nn.Conv2d(256, 512, (3, 3), padding=1, bias=False)

            def forward(self, x):
                x = self.conv1(x)
                x = F.relu(x)
                x = self.conv2(x)
                x = F.relu(x)
                return x

        model = Model().cuda().eval()
        calib_data = [(torch.randn(1, 1, 20, 20).cuda(), ), ]
        cfg = Config()
        cfg.optimization_pipeline = "TensorRT"
        cfg.quantization_calibration_data = calib_data
        cfg.enable_int8 = True
        with cfg:
            opt_model = optimize(model, True, calib_data[0])
        opt_output = opt_model(*calib_data[0])
        model_path = os.path.join(self.tmp_dir.name, "int8_ptq.pt")
        torch.jit.save(opt_model, model_path)
        opt_model = torch.jit.load(model_path)
        now_opt_output = opt_model(*calib_data[0])
        self.assertTrue(torch.equal(opt_output, now_opt_output))


if __name__ == "__main__":
    unittest.main()

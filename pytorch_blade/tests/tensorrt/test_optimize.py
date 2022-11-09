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

import torch
import torch.nn as nn
import torch.nn.functional as F
from tests.quantization import QuantizationTestCase
from tests.tensorrt import skipIfNoTensorRT
from torch.testing import FileCheck
from torch_blade import optimization as opt
from torch_blade import optimize, utils
from torch_blade.config import Config
from torch_blade.testing.common_utils import TestCase


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        # The channel should be a bit larger to make sure that
        # the fp16 kernel is faster that fp32 and thus be chosen
        # when building fp16 trt engine.
        self.conv1 = nn.Conv2d(4, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 4, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(4)

    def forward(self, x):
        y = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = y + x
        return x


@skipIfNoTensorRT()
class TestOptimize(TestCase):
    def setUp(self):
        super().setUp()
        model = TestModel().cuda().eval()
        dummy_input = torch.randn(1, 4, 64, 64).cuda()
        original_output = model(dummy_input)
        self.model = model
        self.dummy_input = dummy_input
        self.original_output = original_output
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def _calculate_model_output(
            self, model, allow_tracing=None, model_inputs=None
    ):
        new_cfg = Config.get_current_context_or_new()
        new_cfg.optimization_pipeline = "TensorRT"
        if model_inputs is None:
            # used for recording the shape, or the model will not be compiled
            # by trt.
            model_inputs = self.dummy_input
        with new_cfg:
            optimized_model = optimize(model, allow_tracing, model_inputs)
        return optimized_model(self.dummy_input)

    def test_different_allow_tracing(self):
        all_trace_output = self._calculate_model_output(
            self.model, True, (self.dummy_input,)
        )
        all_script_output = self._calculate_model_output(self.model)
        partial_trace_output = self._calculate_model_output(
            self.model, ["layer2", "layer3"], (self.dummy_input,)
        )

        self.assertEqual(self.original_output, all_trace_output)
        self.assertEqual(self.original_output, all_script_output)
        self.assertEqual(self.original_output, partial_trace_output)

    @unittest.skipIf(
        not torch.distributed.is_available(), "torch.distributed is not available"
    )
    def test_different_parallel_model(self):
        dp_model = torch.nn.DataParallel(self.model)
        dp_output = self._calculate_model_output(dp_model)
        self.assertEqual(self.original_output, dp_output)

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl",
                rank=0,
                world_size=1,
                init_method="tcp://127.0.0.1:64752",
            )
        ddp_model = torch.nn.parallel.DistributedDataParallel(self.model)
        ddp_output = self._calculate_model_output(ddp_model)
        self.assertEqual(self.original_output, ddp_output)

    def test_fp16_optimization(self):
        new_cfg = Config.get_current_context_or_new().clone()
        new_cfg.enable_fp16 = True
        with new_cfg:
            fp16_output = self._calculate_model_output(self.model)
            self.assertEqual(self.original_output, fp16_output, atol=2e-2, rtol=2e-2)

    def test_int8_optimization(self):
        new_cfg = Config.get_current_context_or_new().clone()
        new_cfg.enable_int8 = True
        new_cfg.quantization_calibration_data = [(self.dummy_input,), (self.dummy_input,)]
        with new_cfg:
            int8_output = self._calculate_model_output(self.model)
            # if this test is unstable, you can release the atol and rtol
            self.assertEqual(self.original_output, int8_output, atol=1e-1, rtol=1e-1)

    def test_fp16_amp_optimization(self):
        new_cfg = Config.get_current_context_or_new().clone()
        new_cfg.enable_fp16 = True
        new_cfg.fp16_fallback_op_ratio = 0.5
        with new_cfg:
            amp_fp16_output = self._calculate_model_output(
                self.model, model_inputs=self.dummy_input
            )
            self.assertEqual(self.original_output, amp_fp16_output, atol=1e-2, rtol=1e-2)

    def test_secondary_optimization(self):
        new_cfg = Config.get_current_context_or_new().clone()
        new_cfg.enable_fp16 = True
        new_cfg.optimization_pipeline = "TensorRT"
        new_cfg.fp16_fallback_op_ratio = 0.1
        with new_cfg:
            amp_fp16_model = optimize(self.model, True, self.dummy_input)
            origin_output = amp_fp16_model._c.forward(self.dummy_input)

        # This is a workaround. If fp16 amp is enabled, optimized model's code
        # will not be updated until we save and reload it.
        # todo: try to figure out the reason.
        model_file = os.path.join(self.tmp_dir.name, "amp_fp16.pt")
        torch.jit.save(amp_fp16_model, model_file)
        amp_fp16_model = torch.jit.load(model_file)
        new_output = amp_fp16_model(self.dummy_input)

        self.assertEqual(origin_output, new_output)
        new_cfg = Config.get_current_context_or_new()
        new_cfg.optimization_pipeline = "TensorRT"
        with new_cfg:
            amp_fp16_fp32_model = optimize(amp_fp16_model, False, self.dummy_input)
        new_output2 = amp_fp16_fp32_model(self.dummy_input)
        self.assertEqual(new_output2, new_output)

    def test_partial_optimization(self):
        class Unsupported(nn.Module):
            def forward(self, x):
                with torch.no_grad():
                    return x

        class Supported(nn.Module):
            def forward(self, x):
                return 3.0 * x + 0.0 + 0.0

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.unsupported = Unsupported()
                # TODO(tianyou.gty): Bug fix.
                # When self.supported is the module listed below, there may be some bugs in trt optimization.
                # The output shape of the model before and after trt optimization is [1, 1] and [1] respectively.
                # self.supported = nn.Sequential(
                #     nn.Linear(1, 1),
                #     nn.Linear(1, 1),
                #     nn.Linear(1, 1)
                # )
                self.supported = Supported()

            def forward(self, x):
                x = self.unsupported(x)
                x = self.supported(x)
                return x

        dummy_input = torch.randn(1).cuda()
        model = Model().cuda().eval()
        origin_output = model(dummy_input)
        cfg = Config.get_current_context_or_new()
        cfg.optimization_pipeline = "TensorRT"
        with cfg:
            if utils.torch_version_number() >= utils.parse_version('1.7.1'):
                optimized_model = optimize(model, False, dummy_input)
                graph = optimized_model.forward.graph
            else:
                optimized_model = optimize(model, ["supported"], dummy_input)
                graph = optimized_model.supported.forward.graph

        optimized_output = optimized_model(dummy_input)
        self.assertEqual(origin_output, optimized_output)
        groups = [n for n in graph.nodes() if n.kind() == "prim::GetAttr"]
        is_trt_engine = [n.s("name").startswith("trt_grp") for n in groups]
        self.assertTrue(any(is_trt_engine))

    def test_convert_onnx_shape_enable(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(4, 20)
                self.linear2 = torch.nn.Linear(4, 20)

            def forward(self, x):
                y = F.relu(self.linear1(x))
                z = y.view([y.size(0), y.size(1), y.size(2) * 5, y.size(3) // 5])
                return F.relu(self.linear2(z))

        model = Net()
        module = torch.jit.script(model).eval()
        inputs = torch.ones(1, 3, 4, 4)
        cfg = Config.get_current_context_or_new()
        cfg.enable_onnx_shape_white_list = True
        cfg.optimization_pipeline = 'TensorRT'
        with cfg:
            opt_module = opt.optimize(
                module,
                model_inputs=inputs,
            )

        expect_gstr = """
graph(%self.1 : __torch__.___torch_mangle_0.Net,
      %x.1 : Float(1:48, 3:16, 4:4, 4:1)):
  # CHECK-NOT: aten::size
  # CHECK-NOT: aten::view
  # CHECK-NOT: aten::mul
  %65 : __torch__.torch.classes.torch_blade.Engine = prim::GetAttr[name="trt_grp0_len26_0"](%self.1)
  %66 : Tensor[] = prim::ListConstruct(%x.1)
  # CHECK: Tensor[] = prim::CallMethod[name="execute"]
  %67 : Tensor[] = prim::CallMethod[name="execute"](%65, %66)
  %69 : Float(1:1200, 3:400, 20:20, 20:1) = prim::ListUnpack(%67)
  return (%69)"""
        FileCheck().run(expect_gstr, opt_module.graph)


@skipIfNoTensorRT()
class TestInt8CalibrationInputTypes(QuantizationTestCase):
    def _do_int8_calibration_optimzie(self, model, calib_data):
        cfg = Config.get_current_context_or_new()
        cfg.optimization_pipeline = "TensorRT"
        cfg.enable_int8 = True
        cfg.quantization_calibration_data = calib_data
        with cfg:
            optimize(model, True, calib_data[0])

    def test_float(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(256, 256, 3, 1, bias=False)
                self.conv2 = nn.Conv2d(256, 256, 3, 1, bias=False)
                self.conv3 = nn.Conv2d(256, 256, 3, 1, bias=False)

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                return x

        model = Model().eval().to(self.device)
        calib_data = [(torch.randn(1, 256, 128, 128).to(self.device), ), ]
        self._do_int8_calibration_optimzie(model, calib_data)

    def test_long(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(256, 256, bias=False)
                self.linear2 = nn.Linear(256, 256, bias=False)
                self.linear3 = nn.Linear(256, 256, bias=False)

            def forward(self, x):
                x = 1.0 * x
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.linear3(x)
                return x
        model = Model().eval().to(self.device)
        calib_data = [(torch.ones(1, 256, dtype=torch.long).to(self.device), ), ]
        self._do_int8_calibration_optimzie(model, calib_data)

    def test_non_contiguous_data(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x + y + x + y

        model = Model().eval().to(self.device)
        contiguous_t = torch.randn(10, 10).to(self.device)
        non_contiguous_t = torch.randn(20, 20).to(self.device)[::2, ::2]
        calib_data = [
            (non_contiguous_t, non_contiguous_t),
            (contiguous_t, contiguous_t)
        ]
        self._do_int8_calibration_optimzie(model, calib_data)


if __name__ == "__main__":
    unittest.main()

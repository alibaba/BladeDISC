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
from torch_blade.config import Config
from torch_blade.testing.common_utils import TestCase
from torch_blade.optimization import optimize
from tests.tensorrt import skipIfNoTensorRT

@skipIfNoTensorRT()
class TestTensorRTEngine(TestCase):

    def _load_model(self, type):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.x = torch.Tensor([0]).to(type).cuda()
                self.y = torch.Tensor([1]).to(type).cuda()
                self.z = torch.Tensor([2]).to(type).cuda()

            def forward(self, w: torch.Tensor):
                return self.x + self.y + self.z + w + w + w

        model = Model().eval()
        return model

    def _check_trt_engine_output(self, opt_model, input, type):
        # check graph output
        output0 = opt_model._c.forward(input)
        self.assertEqual(output0.dtype, type)

        # check trt engine output
        trt_engines = tensorrt.collect_engines(opt_model)
        self.assertEqual(len(trt_engines), 1)
        trt_engine_name, trt_engine = trt_engines[0]
        output1 = trt_engine.execute([input])
        self.assertEqual(len(output1), 1)
        output1 = output1[0]
        self.assertEqual(output0, output1)

    def _test_tensorrt_type_normal(self, model, opt_model, type):
        input = torch.Tensor([4]).to(type).cuda()
        output0 = model(input)
        output1 = opt_model._c.forward(input)
        self.assertEqual(output0, output1)
        self.assertEqual(output0.dtype, output1.dtype)
        self._check_trt_engine_output(opt_model, input, type)

    def _test_tensorrt_type_overflow(self, model, opt_model, type):
        input = torch.Tensor([1e10]).to(type).cuda()
        output0 = model(input)
        output1 = opt_model._c.forward(input)
        self.assertFalse((output0 == output1).item())
        self.assertEqual(output0.dtype, output1.dtype)
        self._check_trt_engine_output(opt_model, input, type)

    def _test_tensorrt_type_cast(self, type):
        input = torch.Tensor([3]).to(type).cuda()
        model = self._load_model(type)
        cfg = Config.get_current_context_or_new()
        cfg.optimization_pipeline = tensorrt.backend_name()
        with cfg:
            opt_model = optimize(model, allow_tracing=True, model_inputs=input)
        self._test_tensorrt_type_normal(model, opt_model, type)
        self._test_tensorrt_type_overflow(model, opt_model, type)

    def test_tensorrt_type_cast_long(self):
        self._test_tensorrt_type_cast(torch.long)

    def test_no_output_overwrite(self):
        class Triple(torch.nn.Module):
            def forward(self, x):
                return 3. * x + 0. + 0.

        x = torch.randn(1)
        cfg = Config.get_current_context_or_new()
        cfg.optimization_pipeline = tensorrt.backend_name()
        with cfg:
            triple = optimize(Triple().eval(), allow_tracing=True, model_inputs=x)

        one = torch.tensor([1], dtype=torch.float).cuda()
        two = torch.tensor([2], dtype=torch.float).cuda()

        three = triple(one)
        self.assertEqual(three, 3 * one)

        six = triple(two)
        self.assertEqual(six, 3 * two)

        self.assertEqual(three, 3 * one)

    @unittest.skipIf(torch.cuda.is_available() and torch.cuda.get_device_capability() <= (6, 1), "TensorRT precision error")
    def test_empty_onnx_export(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.linear = torch.nn.Linear(2, 2)
                self.dropout = torch.nn.Dropout(p=0.8)

            def forward(self, x):
                x = self.linear(x)
                x = self.dropout(x)
                return x.t().contiguous().detach().view(1, 4)

        one = torch.ones([2, 2], dtype=torch.float).cuda()
        model = Model().cuda().eval()
        out = model(one)
        cfg = Config.get_current_context_or_new()
        cfg.optimization_pipeline = tensorrt.backend_name()
        with cfg:
            opt = optimize(model, allow_tracing=True, model_inputs=one)
        self.assertEqual(tensorrt.num_engines(opt), 1)
        self.assertEqual(tensorrt.num_engines(opt), len(tensorrt.num_compiled_nodes(opt)))

        opt_out = opt(one)
        self.assertEqual(out, opt_out)

    def test_acc_verify(self):
        type = torch.float
        input = torch.Tensor([3]).to(type).cuda()
        model = self._load_model(type)
        cfg = Config.get_current_context_or_new()
        cfg.optimization_pipeline = tensorrt.backend_name()
        with cfg:
            opt_model = optimize(model, allow_tracing=True, model_inputs=input)
        trt_engines = tensorrt.collect_engines(opt_model)
        self.assertEqual(len(trt_engines), 1)
        _, trt_engine = trt_engines[0]
        if hasattr(opt_model, 'BladeInit'):
            opt_model.BladeInit.forward()

        with tools.record_cluster_io_context():
            inputs = [input]
            outputs = trt_engine.execute(inputs)
            last_inps = trt_engine.last_inputs()
            last_outs = trt_engine.last_outputs()
            self.assertEqual(inputs, last_inps)
            self.assertEqual(outputs, last_outs)

        trt_engine.execute(inputs)
        with self.assertRaisesRegex(RuntimeError, "Only avaliable with RecordClusterIOFlag set"):
            last_inps = trt_engine.last_inputs()

        with self.assertRaisesRegex(RuntimeError, "Only avaliable with RecordClusterIOFlag set"):
            last_outs = trt_engine.last_outputs()

if __name__ == '__main__':
    unittest.main()

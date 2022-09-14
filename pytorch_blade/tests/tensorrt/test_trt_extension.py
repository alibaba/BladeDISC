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

import io
import unittest
import tempfile
import torch
import torch_blade._torch_blade._backends as backends

from torch_blade import tensorrt
from torch_blade.testing.common_utils import Feedforward, TestCase
from tests.tensorrt import skipIfNoTensorRT


@skipIfNoTensorRT()
class TestTrtExtension(TestCase):
    def setUp(self):
        super().setUp()
        model = Feedforward(16, 32).cuda().eval()
        dummy_input = torch.randn(1, 3, 16, 16).cuda()
        original_output = model(dummy_input)

        self.model = model
        self.dummy_input = dummy_input
        self.origin_output = original_output

    def test_trt_engine_extension(self):
        class Model(torch.nn.Module):
            def __init__(self, state):
                super().__init__()
                self._trt_engine_ext = backends.create_engine(state)

            def forward(self, x):
                return self._trt_engine_ext.execute([x])[0]

        input_names = ["input"]
        output_names = ["output"]
        test_data = [self.dummy_input]
        with io.BytesIO() as onnx_proto_f:
            torch.onnx.export(
                self.model,
                tuple(test_data),
                onnx_proto_f,
                input_names=input_names,
                output_names=output_names,
            )
            onnx_proto = onnx_proto_f.getvalue()
        self.assertTrue(tensorrt.is_onnx2trt_supported(onnx_proto))

        def _copy_meta(data, name, sizes):
            data.name = name
            data.dtype = "Float"
            data.sizes = sizes
            return data

        state = backends.EngineState()
        state.inputs = [
            _copy_meta(backends.TensorInfo(), name, list(tensor.shape))
            for name, tensor in zip(input_names, test_data)
        ]
        state.outputs = [
            _copy_meta(backends.TensorInfo(), name, []) for name in output_names
        ]

        # try to convert onnx to tensorrt engine
        state = tensorrt.cvt_onnx_to_tensorrt(onnx_proto, state, [])
        self.assertEqual(state.backend_name, "TensorRT")

        engine = Model(state)
        ext_output = engine(self.dummy_input)
        self.assertEqual(self.origin_output, ext_output)

        engine = torch.jit.script(engine)
        with tempfile.NamedTemporaryFile() as fp:
            torch.jit.save(engine, fp.name)
            engine = torch.jit.load(fp.name)
        ext_output = engine(self.dummy_input)
        self.assertEqual(self.origin_output, ext_output)


if __name__ == "__main__":
    unittest.main()

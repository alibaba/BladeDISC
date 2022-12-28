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
from tests.neural_engine.testing_base import NeuralEngineTestCase
from torch import nn
from torch.nn import functional as F
from torch_blade import tools
from torch_blade.config import Config
from torch_blade.neural_engine.neural_engine_optimization import \
    _get_unsupported_nodes


class TestSingleOpSupportInfo(NeuralEngineTestCase):
    def _test_single_layer(self, model, inp):
        traced_model = torch.jit.trace(model, inp)
        traced_model = tools.freeze_module(
            traced_model._c, disableShapePeephole=False
        )
        graph = traced_model.forward.graph
        unsupported = _get_unsupported_nodes(graph)
        self.assertEqual(len(unsupported), 0)

    def test_add(self):
        class Model(nn.Module):
            def forward(self, x, y):
                return x + y
        model = Model().eval()
        inp = (torch.randn(5), torch.randn(5))
        self._test_single_layer(model, inp)

    def test_softmax(self):
        class Model(nn.Module):
            def forward(self, x):
                x = F.softmax(x)
                return x
        model = Model().eval()
        inp = torch.randn(5)
        self._test_single_layer(model, inp)

    def test_slice(self):
        class Model(nn.Module):
            def forward(self, x):
                return x[1:]
        model = Model().eval()
        inp = torch.randn(5)
        self._test_single_layer(model, inp)

    def test_reduce_mean(self):
        class Model(nn.Module):
            def forward(self, x):
                return x.mean()

        model = Model().eval()
        inp = torch.randn(5)
        self._test_single_layer(model, inp)

    def test_reshape(self):
        class Model(nn.Module):
            def forward(self, x):
                return x.view(-1)

        model = Model().eval()
        inp = torch.randn(1, 5)
        cfg = Config.get_current_context_or_new()
        cfg.enable_onnx_shape_white_list = False
        with cfg:
            self._test_single_layer(model, inp)

    def test_concat(self):
        class Model(nn.Module):
            def forward(self, x):
                return torch.cat([x, x], dim=0)

        model = Model().eval()
        inp = torch.randn(1, 5)
        self._test_single_layer(model, inp)

    def test_quantize_dequantize(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = 0.1
                self.zero_point = 0
                self.quant_min = 0
                self.quant_max = 255

            def forward(self, x):
                x = torch.fake_quantize_per_tensor_affine(
                    x, self.scale, self.zero_point,
                    self.quant_min, self.quant_max
                )
                return x
        # u8 per-tensor for activation
        model = Model().eval()
        inp = torch.randn(2, 3)
        self._test_single_layer(model, inp)

    def test_transpose(self):
        class Model(nn.Module):
            def forward(self, x):
                return x.transpose(1, 0)
        model = Model().eval()
        inp = torch.randn(2, 3)
        self._test_single_layer(model, inp)

    def test_matmul(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 3, bias=False)

            def forward(self, x):
                return self.linear(x)

        model = Model().eval()
        # test 2d
        inp = torch.randn(1, 2)
        self._test_single_layer(model, inp)

        # test 3d
        inp = torch.randn(1, 1, 2)
        self._test_single_layer(model, inp)

    def test_sqrt(self):
        class Model(nn.Module):
            def forward(self, x):
                return torch.sqrt(x)
        model = Model().eval()
        inp = torch.randn(2, 3)
        self._test_single_layer(model, inp)

    def test_unsqueeze(self):
        class Model(nn.Module):
            def forward(self, x):
                return torch.unsqueeze(x, 0)
        model = Model().eval()
        inp = torch.randn(2, 3)
        self._test_single_layer(model, inp)

    def test_pow(self):
        class Model(nn.Module):
            def forward(self, x):
                return torch.pow(x, 2)
        model = Model().eval()
        inp = torch.randn(2)
        self._test_single_layer(model, inp)

    def test_tanh(self):
        class Model(nn.Module):
            def forward(self, x):
                return torch.tanh(x)
        model = Model().eval()
        inp = torch.randn(2)
        self._test_single_layer(model, inp)

    def test_div(self):
        class Model(nn.Module):
            def forward(self, x):
                return torch.div(x, 2.0)
        model = Model().eval()
        inp = torch.randn(2)
        self._test_single_layer(model, inp)

    def test_mul(self):
        class Model(nn.Module):
            def forward(self, x):
                return x * 2.0
        model = Model().eval()
        inp = torch.randn(2)
        cfg = Config.get_current_context_or_new()
        cfg.enable_onnx_shape_white_list = False
        with cfg:
            self._test_single_layer(model, inp)

    def test_sub(self):
        class Model(nn.Module):
            def forward(self, x):
                return x - 2.0
        model = Model().eval()
        inp = torch.randn(2)
        self._test_single_layer(model, inp)

    def test_relu(self):
        class Model(nn.Module):
            def forward(self, x):
                return F.relu(x)
        model = Model().eval()
        inp = torch.randn(2)
        self._test_single_layer(model, inp)

    def test_conv(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(2, 3, 1)

            def forward(self, x):
                return self.conv(x)
        model = Model().eval()
        inp = torch.randn(1, 2, 5, 5)
        self._test_single_layer(model, inp)


if __name__ == "__main__":
    unittest.main()

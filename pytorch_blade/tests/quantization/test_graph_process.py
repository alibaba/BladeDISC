# Copyright 2021 The BladeDISC Authors. All rights reserved.
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
from tests.quantization import (
    TORCH_VERSION,
    ModelWithFakeQuant,
    PerChannelFakeQuant,
    PerTensorFakeQuant,
    QuantizationTestCase,
    zero_point_dtype
)
from torch import nn

try:
    from torch.ao.quantization.observer import PerChannelMinMaxObserver
except ModuleNotFoundError:
    from torch.quantization.observer import PerChannelMinMaxObserver

from torch.nn import functional as F
from torch.testing import FileCheck
from torch_blade import tools
from torch_blade.quantization import (
    _jit_add_fake_quant_for_weight,
    _jit_pass_add_placeholder_for_fake_quant,
    _jit_pass_remove_all_placeholder,
    _jit_replace_aten_fake_quant_with_custom_version,
    get_fake_quant_node
)


class TestInsertRemoveFakeQuant(QuantizationTestCase):
    def test_insert_and_remove_fake_quant(self):
        model = ModelWithFakeQuant().to(self.device)
        inp = torch.randn(1, 4, 5, 5, device=self.device)
        traced_model = torch.jit.trace(model, inp)
        c_module = traced_model._c
        _jit_pass_add_placeholder_for_fake_quant(c_module)
        graph = c_module.forward.graph
        graph_with_placeholder_str = """graph(%self : __torch__.tests.quantization.ModelWithFakeQuant,
            %x : Tensor):
              # CHECK: aten::fake_quantize_per_tensor_affine
              Tensor = aten::fake_quantize_per_tensor_affine
              # CHECK: aten::fake_quantize_per_channel_affine
              Tensor = aten::fake_quantize_per_channel_affine
              # CHECK: torch_blade::placeholder
              Tensor = torch_blade::placeholder
              # CHECK: aten::_convolution
              Tensor = aten::_convolution
              return

        """
        FileCheck().run(graph_with_placeholder_str, graph)

        # test fake_quant of weight will not be folded by
        # constant propagation
        torch._C._jit_pass_constant_propagation(graph)
        FileCheck().run(graph_with_placeholder_str, graph)

        _jit_pass_remove_all_placeholder(c_module)
        graph_without_placeholder_str = """graph(%self : __torch__.tests.quantization.ModelWithFakeQuant,
                    %x : Tensor):
                      # CHECK: aten::fake_quantize_per_tensor_affine
                      Tensor = aten::fake_quantize_per_tensor_affine
                      # CHECK: aten::fake_quantize_per_channel_affine
                      Tensor = aten::fake_quantize_per_channel_affine
                      # CHECK-NOT: torch_blade::placeholder
                      Tensor = torch_blade::placeholder
                      # CHECK: aten::_convolution
                      Tensor = aten::_convolution
                      return

                """
        graph = c_module.forward.graph
        FileCheck().run(graph_without_placeholder_str, graph)


class TestReplaceAtenFakeQuant(QuantizationTestCase):
    def _test_replace_aten_fake_quant(self, model, inp, all_target_val):
        traced_model = torch.jit.trace(model, inp)
        origin_output = traced_model(inp)
        c_module = traced_model._c
        _jit_replace_aten_fake_quant_with_custom_version(c_module)
        _jit_pass_add_placeholder_for_fake_quant(c_module)
        # to make it easy to get each quantization value
        c_module = tools.freeze_module(c_module, [], disableShapePeephole=False)
        graph = c_module.forward.graph
        fake_quant_nodes = get_fake_quant_node(graph)
        self.assertEqual(len(fake_quant_nodes), len(all_target_val))
        for n, node_target_val in zip(fake_quant_nodes, all_target_val):
            self._test_fake_quant_params(n, node_target_val)

        # test the output is unchanged after the fake_quant op is replaced
        c_module.create_method_from_graph("_forward", graph)
        now_output = c_module._forward(inp)
        self.assertTrue(torch.equal(origin_output, now_output))

    def test_per_tensor_symmetry(self):
        inp = torch.randn(1, 3).to(self.device)
        test_bit_list = [8, 32]
        for bit in test_bit_list:
            # per-tensor + symmetry quantization
            model = PerTensorFakeQuant(
                scale=0.1,
                zero_point=0,
                quant_min=-2 ** (bit - 1),
                quant_max=2 ** (bit - 1) - 1
            ).eval().to(self.device)

            target_val = {
                "scale": torch.tensor(model.scale, device=self.device),
                "zero_point": torch.tensor(model.zero_point, dtype=torch.int, device=self.device),
                "quant_min": model.quant_min,
                "quant_max": model.quant_max,
                "num_bits": bit,
                "use_signed": True,
                "use_symmetric": True,
                "use_dynamic": False,
                "use_per_channel": False
            }
            self._test_replace_aten_fake_quant(model, inp, [target_val, ])

    def test_per_tensor_asymmetry(self):
        inp = torch.randn(1, 3).to(self.device)
        test_bit_list = [8, 32]
        for bit in test_bit_list:
            # per-tensor + asymmetry quantization
            model = PerTensorFakeQuant(
                scale=0.1,
                zero_point=33,
                quant_min=0,
                quant_max=2**bit - 1
            ).eval().to(self.device)
            target_val = {
                "scale": torch.tensor(model.scale, device=self.device),
                "zero_point": torch.tensor(model.zero_point, dtype=torch.int, device=self.device),
                "quant_min": model.quant_min,
                "quant_max": model.quant_max,
                "num_bits": bit,
                "use_signed": False,
                "use_symmetric": False,
                "use_dynamic": False,
                "use_per_channel": False
            }
            self._test_replace_aten_fake_quant(model, inp, [target_val, ])

    def test_per_channel_symmetry(self):
        inp = torch.randn(1, 3).to(self.device)
        # per-channel + symmetry quantization
        scale = torch.tensor([0.1, 0.2, 0.3]).to(self.device)
        zero_point = torch.tensor([0, 0, 0], dtype=zero_point_dtype).to(self.device)
        test_bit_list = [8, 32]
        for bit in test_bit_list:
            model = PerChannelFakeQuant(
                scale=scale,
                zero_point=zero_point,
                quant_min=-2 ** (bit - 1),
                quant_max=2 ** (bit - 1) - 1,
                axis=1
            ).eval().to(self.device)
            target_val = {
                "scale": model.scale,
                "zero_point": model.zero_point,
                "quant_min": model.quant_min,
                "quant_max": model.quant_max,
                "num_bits": bit,
                "use_signed": True,
                "use_symmetric": True,
                "use_dynamic": False,
                "use_per_channel": True
            }
            self._test_replace_aten_fake_quant(model, inp, [target_val, ])

    def test_per_channel_asymmetry(self):
        inp = torch.randn(1, 3).to(self.device)
        # per-channel + asymmetry quantization
        scale = torch.tensor([0.1, 0.2, 0.3]).to(self.device)
        zero_point = torch.tensor([11, 22, 33], dtype=zero_point_dtype).to(self.device)
        test_bit_list = [8, 32]
        for bit in test_bit_list:
            model = PerChannelFakeQuant(
                scale=scale,
                zero_point=zero_point,
                quant_min=0,
                quant_max=2 ** bit - 1,
                axis=1
            ).eval().to(self.device)
            target_val = {
                "scale": model.scale,
                "zero_point": model.zero_point,
                "quant_min": model.quant_min,
                "quant_max": model.quant_max,
                "num_bits": bit,
                "use_signed": False,
                "use_symmetric": False,
                "use_dynamic": False,
                "use_per_channel": True
            }
            self._test_replace_aten_fake_quant(model, inp, [target_val, ])

    def test_dummy_model(self):
        # test dummy model
        inp = torch.randn(1, 4, 5, 5, device=self.device)
        model = ModelWithFakeQuant().eval().to(self.device)
        inp_target_val = {
            "scale": torch.tensor(model.input_scale, device=self.device),
            "zero_point": torch.tensor(model.input_zero_point, dtype=torch.int, device=self.device),
            "quant_min": 0,
            "quant_max": 255,
            "num_bits": 8,
            "use_signed": False,
            "use_symmetric": False,
            "use_dynamic": False,
            "use_per_channel": False
        }
        weight_target_val = {
            "scale": model.weight_scale.data,
            "zero_point": model.weight_zero_point.data.to(zero_point_dtype),
            "quant_min": model.weight_quant_min,
            "quant_max": model.weight_quant_max,
            "num_bits": 8,
            "use_signed": True,
            "use_symmetric": True,
            "use_dynamic": False,
            "use_per_channel": True
        }
        target_val = [inp_target_val, weight_target_val]
        self._test_replace_aten_fake_quant(model, inp, target_val)


class TestAddFakeQuantForWeight(QuantizationTestCase):
    def _test_add_fake_quant_for_weight(self, model, inp, target_quant_info, target_output, target_graph):
        traced_model = torch.jit.trace(model, inp)
        c_module = traced_model._c
        c_module = tools.freeze_module(c_module, [], disableShapePeephole=False)
        _jit_add_fake_quant_for_weight(c_module)
        graph = c_module.forward.graph
        fake_quant_nodes = get_fake_quant_node(graph)
        self.assertEqual(len(fake_quant_nodes), len(target_quant_info))
        for n, node_target_val in zip(fake_quant_nodes, target_quant_info):
            self._test_fake_quant_params(n, node_target_val)

        c_module.create_method_from_graph("_forward", graph)
        now_output = c_module._forward(inp)
        self.assertEqual(now_output, target_output)

        FileCheck().run(target_graph, graph)

    @unittest.skipIf(TORCH_VERSION < (1, 9), "Unsupported torch version")
    def test_per_channel_symmetry(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 4, bias=True)

            def forward(self, x):
                x = self.linear(x)
                return x

        inp = torch.randn(1, 3)
        model = Model().eval()
        # use torch's observer to calculate the scale & zero point
        obs = PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_channel_symmetric
        )
        observer = obs()
        observer(model.linear.weight)
        scale, zero_point = observer.calculate_qparams()
        zero_point = zero_point.to(zero_point_dtype)
        fake_quantized_weight = torch.fake_quantize_per_channel_affine(
            model.linear.weight, scale, zero_point, 0, -128, 127)
        target_output = F.linear(inp, fake_quantized_weight, model.linear.bias)

        target_quant_info = [{
            "scale": scale,
            "zero_point": zero_point,
            "quant_min": -128,
            "quant_max": 127,
            "num_bits": 8,
            "use_signed": True,
            "use_symmetric": True,
            "use_dynamic": True,
            "use_per_channel": True,
            "target_output": target_output
        }]

        target_graph = """
graph(%self.1 : __torch__.___torch_mangle_2.Model,
      %x : Float(1, 3, strides=[3, 1], requires_grad=0, device=cpu)):
  %self.linear.weight : Float(4, 3, strides=[3, 1], requires_grad=0, device=cpu) = prim::Constant[value=<Tensor>]()
  %11 : Float(4, strides=[1], requires_grad=0, device=cpu) = prim::Constant[value=0.001 *  2.5694  2.9053  1.9990  2.3660 [ CPUFloatType{4} ]]()
  %12 : Int(4, strides=[1], requires_grad=0, device=cpu) = prim::Constant[value= 0  0  0  0 [ CPUIntType{4} ]]()
  %13 : int = prim::Constant[value=-128]()
  %14 : int = prim::Constant[value=127]()
  %15 : int = prim::Constant[value=8]()
  %16 : int = prim::Constant[value=1]()
  %17 : int[] = prim::ListConstruct(%16)
  %18 : bool = prim::Constant[value=1]()
  %19 : bool = prim::Constant[value=1]()
  %20 : bool = prim::Constant[value=1]()
  %21 : bool = prim::Constant[value=1]()
  # CHECK: torch_blade::fake_quant
  %10 : Float(4, 3, strides=[3, 1], requires_grad=0, device=cpu) = torch_blade::fake_quant(%self.linear.weight, %11, %12, %13, %14, %15, %17, %18, %19, %20, %21)
  %self.linear.bias : Float(4, strides=[1], requires_grad=0, device=cpu) = prim::Constant[value= 0.1169 -0.2259 -0.2832  0.1494 [ CPUFloatType{4} ]]()
  %6 : Float(1, 4, strides=[4, 1], requires_grad=1, device=cpu) = aten::linear(%x, %10, %self.linear.bias)
  return (%6)
"""

        self._test_add_fake_quant_for_weight(model, inp, target_quant_info, target_output, target_graph)


if __name__ == "__main__":
    unittest.main()

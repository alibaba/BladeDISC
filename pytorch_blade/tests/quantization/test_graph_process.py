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
    ModelWithFakeQuant,
    PerChannelFakeQuant,
    PerTensorFakeQuant,
    QuantizationTestCase,
    zero_point_dtype
)
from torch.testing import FileCheck
from torch_blade import tools
from torch_blade.quantization import (
    _jit_pass_add_placeholder_for_fake_quant,
    _jit_pass_remove_all_placeholder,
    _jit_replace_aten_fake_quant_with_custom_version,
    _quantization
)


def _get_fake_quant_node(graph):
    fake_quant_nodes = []
    for n in graph.nodes():
        if n.kind() == _quantization.torch_blade_fake_quant_name:
            fake_quant_nodes.append(n)
    return fake_quant_nodes


class TestGraphProcess(QuantizationTestCase):
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

    def _test_fake_quant_params(self, fake_quant_node, target_val):
        # The order of the constant nodes should not be fixed. So it is not easy to
        # use the FileCheck system to check each attributes of the fake_quant node.
        # We extract all attributes and compare them with the target value one-by-one.
        input_list = fake_quant_node.input_list()
        scale = input_list[1].node().t("value")
        self.assertTrue(torch.equal(scale, target_val['scale']))

        zero_point = input_list[2].node().t("value")
        self.assertTrue(torch.equal(zero_point, target_val['zero_point']))

        quant_min = input_list[3].node().i("value")
        self.assertEqual(quant_min, target_val["quant_min"])

        quant_max = input_list[4].node().i("value")
        self.assertEqual(quant_max, target_val["quant_max"])

        num_bits = input_list[5].node().i("value")
        self.assertEqual(num_bits, target_val["num_bits"])

        # TODO: find a way to check axis

        use_signed = bool(input_list[7].node().i("value"))
        self.assertEqual(use_signed, target_val["use_signed"])

        use_symmetric = bool(input_list[8].node().i("value"))
        self.assertEqual(use_symmetric, target_val["use_symmetric"])

        use_dynamic = bool(input_list[9].node().i("value"))
        self.assertEqual(use_dynamic, target_val["use_dynamic"])

        use_per_channel = bool(input_list[10].node().i("value"))
        self.assertEqual(use_per_channel, target_val["use_per_channel"])

    def _test_replace_aten_fake_quant(self, model, inp, all_target_val):
        traced_model = torch.jit.trace(model, inp)
        origin_output = traced_model(inp)
        c_module = traced_model._c
        _jit_replace_aten_fake_quant_with_custom_version(c_module)
        _jit_pass_add_placeholder_for_fake_quant(c_module)
        # to make it easy to get each quantization value
        c_module = tools.freeze_module(c_module, [], disableShapePeephole=False)
        graph = c_module.forward.graph
        fake_quant_nodes = _get_fake_quant_node(graph)
        self.assertEqual(len(fake_quant_nodes), len(all_target_val))
        for n, node_target_val in zip(fake_quant_nodes, all_target_val):
            self._test_fake_quant_params(n, node_target_val)

        # test the output is unchanged after the fake_quant op is replaced
        c_module.create_method_from_graph("_forward", graph)
        now_output = c_module._forward(inp)
        self.assertTrue(torch.equal(origin_output, now_output))

    def test_replace_aten_fake_quant(self):
        inp = torch.randn(1, 3).to(self.device)

        # per-tensor + symmetry quantization
        model = PerTensorFakeQuant(
            scale=0.1,
            zero_point=0,
            quant_min=-128,
            quant_max=127
        ).eval().to(self.device)

        target_val = {
            "scale": torch.tensor(model.scale, device=self.device),
            "zero_point": torch.tensor(model.zero_point, dtype=torch.int, device=self.device),
            "quant_min": model.quant_min,
            "quant_max": model.quant_max,
            "num_bits": 8,
            "use_signed": True,
            "use_symmetric": True,
            "use_dynamic": False,
            "use_per_channel": False
        }
        self._test_replace_aten_fake_quant(model, inp, [target_val, ])

        # per-tensor + asymmetry quantization
        model = PerTensorFakeQuant(
            scale=0.1,
            zero_point=33,
            quant_min=0,
            quant_max=255
        ).eval().to(self.device)
        target_val = {
            "scale": torch.tensor(model.scale, device=self.device),
            "zero_point": torch.tensor(model.zero_point, dtype=torch.int, device=self.device),
            "quant_min": model.quant_min,
            "quant_max": model.quant_max,
            "num_bits": 8,
            "use_signed": False,
            "use_symmetric": False,
            "use_dynamic": False,
            "use_per_channel": False
        }
        self._test_replace_aten_fake_quant(model, inp, [target_val, ])

        # per-channel + symmetry quantization
        scale = torch.tensor([0.1, 0.2, 0.3]).to(self.device)
        zero_point = torch.tensor([0, 0, 0], dtype=zero_point_dtype).to(self.device)
        model = PerChannelFakeQuant(
            scale=scale,
            zero_point=zero_point,
            quant_min=-128,
            quant_max=127,
            axis=1
        ).eval().to(self.device)
        target_val = {
            "scale": model.scale,
            "zero_point": model.zero_point,
            "quant_min": model.quant_min,
            "quant_max": model.quant_max,
            "num_bits": 8,
            "use_signed": True,
            "use_symmetric": True,
            "use_dynamic": False,
            "use_per_channel": True
        }
        self._test_replace_aten_fake_quant(model, inp, [target_val, ])

        # per-channel + asymmetry quantization
        scale = torch.tensor([0.1, 0.2, 0.3]).to(self.device)
        zero_point = torch.tensor([11, 22, 33], dtype=zero_point_dtype).to(self.device)
        model = PerChannelFakeQuant(
            scale=scale,
            zero_point=zero_point,
            quant_min=0,
            quant_max=255,
            axis=1
        ).eval().to(self.device)
        target_val = {
            "scale": model.scale,
            "zero_point": model.zero_point,
            "quant_min": model.quant_min,
            "quant_max": model.quant_max,
            "num_bits": 8,
            "use_signed": False,
            "use_symmetric": False,
            "use_dynamic": False,
            "use_per_channel": True
        }
        self._test_replace_aten_fake_quant(model, inp, [target_val, ])

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


if __name__ == "__main__":
    unittest.main()

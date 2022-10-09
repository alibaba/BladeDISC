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
from tests.quantization import ModelWithFakeQuant, QuantizationTestCase
from torch.testing import FileCheck
from torch_blade.quantization import (
    _jit_pass_add_placeholder_for_fake_quant,
    _jit_pass_remove_all_placeholder
)


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


if __name__ == "__main__":
    unittest.main()

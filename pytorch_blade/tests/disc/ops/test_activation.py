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

import torch
import unittest
from unittest import skipIf
from tests.disc.testing_base import skipIfEnableTorchMlir, DiscTestCase, skipTorchLE
from torch_blade import Config
import pytest


class TestDiscActivation(DiscTestCase):
    def _test_activation(self, nn_func=None, native_func=None, dims=[]):
        if nn_func:
            self._test_disc(nn_func, dims)

        if native_func:
            @torch.jit.script
            def jit_script_func(x):
                return native_func(x)

            self._test_disc(jit_script_func, dims)

    def test_relu_dynamic_shape(self):
        self._test_activation(torch.nn.ReLU(), torch.nn.functional.relu, [([-1, -1, -1, -1], torch.float)])

    def test_relu_static_shape(self):
        self._test_activation(torch.nn.ReLU(), torch.nn.functional.relu, [([2, 4, 16, 16], torch.float)])

    def test_leaky_relu_static_shape(self):
        self._test_activation(torch.nn.LeakyReLU(), torch.nn.functional.leaky_relu, [([2, 4, 16, 16], torch.float)])

    def test_silu_static_shape(self):
        self._test_activation(torch.nn.SiLU(), torch.nn.functional.silu, [([2, 4, 16, 16], torch.float)])

    def test_sigmoid_static_shape(self):
        self._test_activation(torch.nn.Sigmoid(), torch.sigmoid, [([2, 4, 16, 16], torch.float)])

    def test_sigmoid_dynamic_shape(self):
        self._test_activation(torch.nn.Sigmoid(), torch.sigmoid, [([-1, -1, -1, -1], torch.float)])

    # @skipTorchLE("1.11.1")
    def test_gelu_static_shape(self):
        self._test_activation(torch.nn.GELU(), torch.nn.functional.gelu, [([2, 4, 16, 16], torch.float)])

    @skipTorchLE("1.11.1")
    def test_gelu_dynamic_shape(self):
        self._test_activation(torch.nn.GELU(), torch.nn.functional.gelu, [([-1, -1, -1, -1], torch.float)])

    def test_hardtanh_static_shape(self):
        self._test_activation(torch.nn.Hardtanh(), torch.nn.functional.hardtanh, [([2, 4, 16, 16], torch.float)])

    def test_hardtanh_dynamic_shape(self):
        self._test_activation(torch.nn.Hardtanh(), torch.nn.functional.hardtanh, [([-1, -1, -1, -1], torch.float)])

    def test_glu(self):
        self._test_activation(torch.nn.GLU(), torch.nn.functional.glu, [([2, 4, 16, 16], torch.float)])

    def test_selu(self):
        self._test_activation(torch.nn.SELU(), torch.nn.functional.selu, [([2, 4, 16, 16], torch.float)])

    def test_hardswish(self):

        def _jit_pass_hardswish(graph):
            from_graph_str = """
            graph(%x):
              %r: Tensor = aten::hardswish(%x)
              return (%r)
            """

            def hard_sigmoid(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
                return torch.nn.functional.relu6(x + 3, inplace) / 6

            @torch.jit.script
            def hard_swish(x: torch.Tensor) -> torch.Tensor:
                return x * hard_sigmoid(x, False)

            torch._C._jit_pass_inline(hard_swish.graph)
            torch._C._jit_pass_dce(hard_swish.graph)
            torch._C._jit_pass_constant_propagation(hard_swish.graph)

            to_graph_str = str(hard_swish.graph)
            torch._C._jit_pass_custom_pattern_based_rewrite_graph(
                from_graph_str, to_graph_str, graph)
            torch._C._jit_pass_dce(graph)
            torch._C._jit_pass_constant_propagation(graph)

        with Config.get_current_context_or_new() as cfg:
            cfg.customize_jit_passes = [_jit_pass_hardswish]
            self._test_activation(None, torch.nn.functional.hardswish, [([2, 4, 16, 16], torch.float)])
            self._test_activation(None, torch.nn.functional.hardswish, [([-1, -1, -1, -1], torch.float)])



if __name__ == "__main__":
    unittest.main()

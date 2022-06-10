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
from tests.disc.testing_base import skipIfNoDISC

from tests.disc.testing_base import DiscTestCase, skipIfEnableTorchMlir
from torch_blade import Config


class TestDiscActivation(DiscTestCase):
    def _test_activation(self, nn_func, native_func):
        x = torch.randn([2, 4, 16, 16], device=self.device)
        test_data = (x, )

        self._test_cvt_to_disc(nn_func, test_data)

        @torch.jit.script
        def jit_func(x):
            return native_func(x)
        self._test_cvt_to_disc(jit_func, test_data)

    def test_relu(self):
        self._test_activation(torch.nn.ReLU(), torch.nn.functional.relu)

    def test_leaky_relu(self):
        self._test_activation(torch.nn.LeakyReLU(), torch.nn.functional.leaky_relu)

    def test_silu(self):
        self._test_activation(torch.nn.SiLU(), torch.nn.functional.silu)

    def test_sigmoid(self):
        self._test_activation(torch.nn.Sigmoid(), torch.nn.functional.sigmoid)

    @skipIfEnableTorchMlir()
    def test_glu(self):
        self._test_activation(torch.nn.GLU(), torch.nn.functional.glu)

    @skipIfEnableTorchMlir()
    def test_gelu(self):
        self._test_activation(torch.nn.GELU(), torch.nn.functional.gelu)

    @skipIfEnableTorchMlir()
    def test_hardtanh(self):
        self._test_activation(torch.nn.Hardtanh(), torch.nn.functional.hardtanh)

    @skipIfEnableTorchMlir()
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

        @torch.jit.script
        def hardswish_func(x):
            return torch.nn.functional.hardswish(x)

        config = Config.get_current_context_or_new()
        config.customize_jit_passes = [_jit_pass_hardswish]
        with config:
            self._test_activation(torch.nn.Hardswish(), torch.nn.functional.hardswish)


if __name__ == "__main__":
    unittest.main()

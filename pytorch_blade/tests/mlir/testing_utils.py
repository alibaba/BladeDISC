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

from torch_blade import mlir
from torch_blade import optimize
from torch_blade.config import Config
from torch_blade.clustering import support_fusion_group
from torch_blade.testing.common_utils import TestCase
from tests.mlir import skipIfNoDISC

@skipIfNoDISC()
class DiscTestCase(TestCase):

    def _ScriptFunction2Module(self, nn_module):
        if isinstance(nn_module, torch.jit.ScriptFunction):
            _compilation_unit = torch._C.CompilationUnit()
            c_module = torch._C.ScriptModule("gen_func", _compilation_unit, True)
            c_module.create_method_from_graph("forward", nn_module.graph)
            return torch.jit._recursive.wrap_cpp_module(c_module)
        else:
            return nn_module

    def cvt_to_disc(self, nn_module, test_data):
        cfg = Config.get_current_context_or_new()
        cfg.optimization_pipeline = "DISC"
        with mlir.testing_context(), support_fusion_group.min_group_nodes(1), cfg:
            nn_module = self._ScriptFunction2Module(nn_module)
            nn_module = nn_module.eval().to(self.device)
            opt_module = optimize(
                nn_module,
                allow_tracing=True,
                model_inputs=test_data,
            )
        return opt_module

    def _test_cvt_to_disc(
        self, nn_module, test_data, rtol=1e-6, atol=1e-3, n_engines=1
    ):
        nn_module = self._ScriptFunction2Module(nn_module)
        nn_module = nn_module.eval().to(self.device)
        result = nn_module(*test_data)
        opt_module = self.cvt_to_disc(nn_module, test_data)
        output = opt_module.forward(*test_data)
        self.assertEqual(output, result, rtol=rtol, atol=atol)
        self.assertGreaterEqual(mlir.num_engines(opt_module), n_engines)
        return output, result

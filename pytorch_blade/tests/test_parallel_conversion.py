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
import torch_blade
from torch_blade.logging import logger_level_context
from torch_blade.clustering.support_fusion_group import min_group_nodes

class TestParallelConversion(unittest.TestCase):

    def test_conversion(self):

        class DimModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                N = 100
                linears = [torch.nn.Linear(10, 10) for _ in range(N)]
                adds = [torch.nn.LeakyReLU() for _ in range(N)]
                interlieaved = list(sum(zip(linears, adds), ()))
                self.layers = torch.nn.Sequential(*interlieaved)

            def forward(self, x):
                return self.layers(x)

        # Original model
        model = DimModel().eval()
        datum = torch.rand(1, 10)
        output = model(datum)

        # Optimize with 100 parallelism
        traced = torch.jit.trace(model, datum)
        cfg = torch_blade.Config.get_current_context_or_new().clone()
        cfg.optimization_pipeline = torch_blade.mlir.backend_name()
        cfg.experimental_subgraph_conversion_parallelism = 100
        cfg.customize_op_black_list = ['aten::relu', 'aten::leaky_relu']
        with cfg, min_group_nodes(1), logger_level_context("DEBUG"):
            opt_model = torch_blade.optimize(traced, model_inputs=(datum,))
        opt_out = opt_model(datum)
        self.assertTrue(torch.allclose(opt_out, output))

if __name__ == "__main__":
    unittest.main()

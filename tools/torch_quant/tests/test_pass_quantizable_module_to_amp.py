# Copyright 2023 The BladeDISC Authors. All rights reserved.
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

from tests.models import SimpleModule
from torch_quant.amp_module import AmpModule
from torch_quant.graph import (
    GraphModContext,
    insert_act_observer,
    insert_w_observer,
    quantizable_module_to_amp,
    set_qconfig,
)
from torch_quant.observer import BiasObserver, MinMaxObserver, PerChannelMinMaxObserver


class QuantizableModuleToAmpTest(unittest.TestCase):
    def test_base(self) -> None:
        model = SimpleModule()
        ctx = GraphModContext(
            gm=torch.fx.symbolic_trace(model),
            root=model,
            act_ob_ctr=MinMaxObserver,
            w_ob_ctr=PerChannelMinMaxObserver,
            bias_ob_ctr=BiasObserver,
        )
        insert_act_observer(ctx)
        insert_w_observer(ctx)
        ctx.gm = torch.fx.symbolic_trace(model)
        set_qconfig(ctx)
        quantizable_module_to_amp(ctx)
        amp_modules = dict(ctx.gm.named_modules())
        for name, mod in model.named_modules():
            if type(mod) in [torch.nn.Conv2d, torch.nn.Linear]:
                self.assertTrue(isinstance(amp_modules[name], AmpModule))

        dummy_input = torch.randn((1, 2, 5, 5))
        original_output = model(dummy_input)
        amp_output = ctx.gm(dummy_input)
        self.assertTrue(torch.equal(original_output, amp_output))


if __name__ == '__main__':
    unittest.main()

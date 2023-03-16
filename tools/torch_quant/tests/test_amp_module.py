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
from functools import partial
from typing import Callable

import torch
from torch.quantization import QConfig

from torch_quant.amp_module import AmpModule
from torch_quant.observed_module import Linear
from torch_quant.observer import (
    BiasObserver,
    MinMaxObserver,
    Observer,
    PerChannelMinMaxObserver,
    toggle_observer,
)


class TestAmpModule(unittest.TestCase):
    def test_basic(self):
        model = torch.nn.Linear(2, 4)
        dummy_input = torch.randn((4, 2))
        original_output = model(dummy_input)

        def _act_ob(data: torch.Tensor) -> None:
            ob = MinMaxObserver()
            ob.set_mode(observe=True, fake_quant=False)
            ob(data)
            return ob

        act_ob = _act_ob(dummy_input)
        out_ob = _act_ob(model(dummy_input))

        def _w_ob(ctr: Callable[..., Observer], param: torch.nn.Parameter) -> Observer:
            ob = ctr()
            ob.set_mode(observe=True, fake_quant=False)
            ob(param)
            return ob

        w_ob_ctr = PerChannelMinMaxObserver
        w_ob = _w_ob(w_ob_ctr, model.weight)
        bias_ob_ctr = partial(BiasObserver, w_ob, act_ob)
        bias_ob = _w_ob(bias_ob_ctr, model.bias)
        model.qconfig = QConfig(activation=None, weight=w_ob_ctr)
        observed_model = Linear.from_float(model, w_ob, bias_ob)
        amp = AmpModule(model, observed_model, act_ob, out_ob)

        amp_output = amp(dummy_input)
        self.assertTrue(torch.equal(original_output, amp_output))

        w_ob.set_mode(observe=False, fake_quant=True)
        quant_weight = w_ob(model.weight)
        model.load_state_dict({'weight': quant_weight, 'bias': model.bias})
        toggle_observer(model, observe=False, fake_quant=True)
        quant_output = out_ob(model(act_ob(dummy_input)))
        mse = torch.mean(torch.pow(original_output - quant_output, 2))
        self.assertTrue(torch.equal(amp.noise, mse))


if __name__ == "__main__":
    unittest.main()

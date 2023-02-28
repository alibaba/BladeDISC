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

from typing import Optional, Type

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.qat as nnqat
from torch_quant.observer import Observer


def _from_float(cls, mod,
                w_ob: Optional[Observer] = None,
                bias_ob: Optional[Observer] = None,) -> None:
    new_mod = super(cls, cls).from_float(mod)
    new_mod.w_ob = w_ob
    new_mod.bias_ob = bias_ob
    delattr(new_mod, 'weight_fake_quant')
    return new_mod


def _conv_type(name: str, base: Type) -> Type:
    def forward(self, input):
        weight = self.w_ob(self.weight) if self.w_ob is not None else self.weight
        bias = self.bias_ob(self.bias) if self.bias_ob is not None else self.bias
        return self._conv_forward(input, weight, bias)

    return type(name, (base,), {'from_float': classmethod(_from_float), 'forward': forward})


Conv1d = _conv_type('Conv1d', nnqat.Conv1d)
Conv2d = _conv_type('Conv2d', nnqat.Conv2d)
Conv3d = _conv_type('Conv3d', nnqat.Conv3d)


def _linear_forward(self, input):
    weight = self.w_ob(self.weight) if self.w_ob is not None else self.weight
    bias = self.bias_ob(self.bias) if self.bias_ob is not None else self.bias
    return F.linear(input, weight, bias)


Linear = type('Linear', (nnqat.Linear, ), {
              'from_float': classmethod(_from_float), 'forward': _linear_forward})

OB_MODULE_MAPPING = {
    nn.Conv1d: Conv1d,
    nn.Conv2d: Conv2d,
    nn.Conv3d: Conv3d,
    nn.Linear: Linear,
}

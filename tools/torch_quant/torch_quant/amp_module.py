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

import logging
from typing import List

import torch
import torch.nn as nn

from torch_quant.observer import Observer, toggle_observer

LOGGER = logging.getLogger(__name__)


class AmpModule(nn.Module):
    """
    This module includes original float op and fake quantized op (i.e. observed module).
    Mean square error is used to analyze the quantization precision.
    """

    def __init__(
        self,
        float_op: nn.Module,
        observed_op: nn.Module,
        act_ob: Observer,
        out_ob: Observer,
    ) -> None:
        super(AmpModule, self).__init__()
        self.float_op = float_op
        self.observed_op = observed_op
        self.act_ob = act_ob
        self.out_ob = out_ob
        self.register_buffer('noise', torch.tensor(0.0))
        toggle_observer(self, observe=False, fake_quant=True)

    def forward(self, x):
        y = self.float_op(x)
        quant_y = self.out_ob(self.observed_op(self.act_ob(x)))
        noise = torch.mean(torch.pow(y.detach() - quant_y.detach(), 2))
        self.noise.copy_(self.noise + noise)
        return y


def get_fallback_names(root: nn.Module, num: int) -> List[str]:
    modules = dict(root.named_modules())
    candidates = [k for k, v in modules.items() if isinstance(v, AmpModule)]
    if len(candidates) < num:
        LOGGER.warning(
            f"No module be quantized. There are only {len(candidates)} "
            f"quantizable modules, but fallback number is {num}."
        )
        num = len(candidates)
    LOGGER.info(f"Fallback {num} modules to float precision.")
    noises = {name: modules[name].noise for name in candidates}
    sorted_noises = sorted(noises.items(), key=lambda x: x[1], reverse=True)
    fallback_names = [k[0] for k in sorted_noises[:num]]
    return fallback_names

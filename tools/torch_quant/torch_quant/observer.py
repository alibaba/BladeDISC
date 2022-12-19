# Copyright 2022 The BladeDISC Authors. All rights reserved.
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
from abc import ABC, abstractmethod
from typing import Any, NamedTuple, Tuple

import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)

# definition of zero_point and scale:
# float_value = (quant_value - zero_point) * scale


class QParams(NamedTuple):
    qscheme: torch.qscheme
    dtype: torch.dtype
    scale: torch.Tensor
    zero_point: torch.Tensor


class Observer(torch.nn.Module, ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.observe = True
        self.fake_quant = True

    @property
    @abstractmethod
    def qparams(self) -> QParams:
        ...


def toggle_observer(root: nn.Module, *, observe: bool, fake_quant: bool) -> None:
    for m in root.modules():
        if isinstance(m, Observer):
            m.observe = observe
            m.fake_quant = fake_quant


DTYPE_TO_BIT_SIGN = {
    torch.qint8: (8, True),
    torch.quint8: (8, False),
}


def calc_quant_min_max(bit: int, signed: bool) -> Tuple[int, int]:
    q_min = - (1 << bit - 1) if signed else 0
    q_max = (1 << bit - 1) - 1 if signed else (1 << bit) - 1
    return q_min, q_max


class MinMaxObserver(Observer):
    def __init__(self, dtype: torch.dtype = torch.qint8,
                 qscheme: torch.qscheme = torch.per_tensor_affine, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dtype = dtype
        self.qscheme = qscheme
        self.bit, self.signed = DTYPE_TO_BIT_SIGN[dtype]
        if self.symmetric and not self.signed:
            raise ValueError('Symmetric quantization requires signed dtype.')
        self.q_min, self.q_max = calc_quant_min_max(self.bit, self.signed)
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.register_buffer("scale", torch.tensor(1.))
        self.register_buffer("zero_point", torch.tensor(0.))

    @property
    def symmetric(self) -> bool:
        return self.qscheme in (torch.per_tensor_symmetric, torch.per_channel_symmetric)

    def _calculate_qparams(self) -> Tuple[Any, Any]:
        if self.symmetric:
            scale = max(max(self.max_val, 0) / self.q_max,
                        min(self.min_val, 0) / self.q_min)
            zero_point = 0.
        else:
            # TODO(litan.ls) aten check zero_point in range of quantized dtype, so we force
            # float val range always cover 0. However, other backends may support zero_point
            # beyond quantized dtype range.
            scale = (max(self.max_val, 0) - min(self.min_val, 0)) / \
                (self.q_max - self.q_min)
            zero_point = self.q_min - min(self.min_val, 0) / scale
        LOGGER.debug(
            f'calc qparams: {self.min_val=}, {self.max_val=}, {self.q_min=}, {self.q_max=}, {self.bit=}, {self.signed=}, {scale=}, {zero_point=}')
        return scale, zero_point

    @property
    def qparams(self) -> QParams:
        return QParams(qscheme=self.qscheme, dtype=self.dtype,
                       scale=self.scale,
                       zero_point=self.zero_point)

    def forward(self, x):
        if self.observe:
            min_val, max_val = torch.aminmax(x.detach().to(self.min_val.dtype))
            self.min_val.copy_(torch.min(min_val, self.min_val))
            self.max_val.copy_(torch.max(max_val, self.max_val))
            scale, zero_point = self._calculate_qparams()
            self.scale.copy_(scale)
            self.zero_point.copy_(zero_point)
        if self.fake_quant:
            return torch.fake_quantize_per_tensor_affine(x, self.scale, self.zero_point, self.q_min, self.q_max)
        else:
            return x

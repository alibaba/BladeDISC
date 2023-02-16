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
from abc import ABC
from functools import partial
from typing import NamedTuple, Tuple

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


def check_min_max_valid(min_val: torch.Tensor, max_val: torch.Tensor) -> bool:
    if min_val.numel() == 0 or max_val.numel() == 0:
        LOGGER.warning("There is no elements in the min_val or max_val, "
                       "please check whether the observer has observed any data.")
        return False

    if min_val.dim() == 0 or max_val.dim() == 0:
        if min_val == float("inf") and max_val == float("-inf"):
            LOGGER.warning(
                "must run observer before calling calculate_qparams. " +
                "Returning default values."
            )

            return False

        assert min_val <= max_val, "min {} should be less than max {}".format(
            min_val, max_val
        )
    else:
        assert torch.all(
            min_val <= max_val
        ), "min {} should be less than max {}".format(min_val, max_val)
    return True


def is_per_channel(qscheme: torch.qscheme):
    return qscheme in (torch.per_tensor_symmetric, torch.per_channel_symmetric)

def pre_load_state_dict_hook(
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs
):
    if module.ch_axis == -1:
        # per-tensor quantization, no extra things should be done
        return
    attr_list = ['scale', 'zero_point', 'min_val', 'max_val']
    for attr in attr_list:
        name = prefix + attr
        if name in state_dict:

            if not hasattr(module, attr):
                raise RuntimeError(f"There is no attribute {attr} for module {prefix},"
                                   f"This may caused by a mismatch between prepared model"
                                   f"with the checkpoint.")
            attr_val_in_module = getattr(module, attr)
            attr_val_in_state_dict = state_dict[name]
            if attr_val_in_module.shape != attr_val_in_state_dict.shape:
                attr_val_in_module.data = torch.ones_like(attr_val_in_state_dict)


class Observer(torch.nn.Module, ABC):
    def __init__(self, dtype, qscheme, ch_axis=-1, **kwargs) -> None:
        super().__init__()
        self.dtype = dtype
        self.qscheme = qscheme
        self.bit, self.signed = DTYPE_TO_BIT_SIGN[dtype]
        if self.symmetric and not self.signed:
            raise ValueError('Symmetric quantization requires signed dtype.')
        self.q_min, self.q_max = calc_quant_min_max(self.bit, self.signed)
        self.register_buffer('eps', torch.tensor([torch.finfo(torch.float32).eps]))
        self.observe = True
        self.fake_quant = True
        self.ch_axis = ch_axis
        self._register_load_state_dict_pre_hook(partial(pre_load_state_dict_hook, self))

    @property
    def symmetric(self) -> bool:
        return is_per_channel(self.qscheme)

    @property
    def per_channel(self) -> bool:
        return self.qscheme in (torch.per_channel_symmetric, torch.per_channel_affine)

    @property
    def qparams(self) -> QParams:
        return QParams(qscheme=self.qscheme, dtype=self.dtype,
                       scale=self.scale,
                       zero_point=self.zero_point)

    def _calculate_qparams(self, min_val, max_val) -> Tuple[torch.Tensor, torch.Tensor]:
        if not check_min_max_valid(min_val, max_val):
            return torch.tensor([1.0], device=min_val.device.type), torch.tensor([0], device=min_val.device.type)

        q_min, q_max = self.q_min, self.q_max
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        device = min_val_neg.device
        scale = torch.ones(min_val_neg.size(), dtype=torch.float32, device=device)
        zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int32, device=device)

        if self.symmetric:
            max_val_pos = torch.max(-min_val_neg, max_val_pos)
            scale = max_val_pos / (float(q_max - q_min) / 2)
            scale = torch.max(scale, self.eps)
            if self.dtype == torch.quint8:
                zero_point = zero_point.new_full(zero_point.size(), 128)
        else:
            scale = (max_val_pos - min_val_neg) / float(q_max - q_min)
            scale = torch.max(scale, self.eps)
            zero_point = q_min - torch.round(min_val_neg / scale).to(torch.int32)
            zero_point = torch.clamp(zero_point, q_min, q_max)

        # For scalar values, cast them to Tensors of size 1 to keep the shape
        # consistent with default values in FakeQuantize.
        if len(scale.shape) == 0:
            # TODO: switch to scale.item() after adding JIT support
            scale = torch.tensor([float(scale)], dtype=scale.dtype, device=device)
        if len(zero_point.shape) == 0:
            # TODO: switch to zero_point.item() after adding JIT support
            zero_point = torch.tensor(
                [int(zero_point)], dtype=zero_point.dtype, device=device
            )
            if self.qscheme == torch.per_channel_affine_float_qparams:
                zero_point = torch.tensor(
                    [float(zero_point)], dtype=zero_point.dtype, device=device
                )

        LOGGER.debug(
            f'calc qparams: {self.min_val=}, {self.max_val=}, {self.q_min=}, {self.q_max=}, {self.bit=}, {self.signed=}, {scale=}, {zero_point=}')
        return scale, zero_point



def toggle_observer(root: nn.Module, *, observe: bool, fake_quant: bool) -> None:
    for m in root.modules():
        if isinstance(m, Observer):
            m.observe = observe
            m.fake_quant = fake_quant


DTYPE_TO_BIT_SIGN = {
    torch.qint8: (8, True),
    torch.quint8: (8, False),
    torch.qint32: (32, True),
}


def calc_quant_min_max(bit: int, signed: bool) -> Tuple[int, int]:
    q_min = - (1 << bit - 1) if signed else 0
    q_max = (1 << bit - 1) - 1 if signed else (1 << bit) - 1
    return q_min, q_max


class MinMaxObserver(Observer):
    def __init__(self, dtype: torch.dtype = torch.qint8,
                 qscheme: torch.qscheme = torch.per_tensor_symmetric, **kwargs) -> None:
        super().__init__(dtype, qscheme, **kwargs)
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.register_buffer("scale", torch.tensor([1.]))
        self.register_buffer("zero_point", torch.tensor([0], dtype=torch.int32))

    def forward(self, x):
        if self.observe:
            min_val, max_val = torch.aminmax(x.detach().to(self.min_val.dtype))
            self.min_val.copy_(torch.min(min_val, self.min_val))
            self.max_val.copy_(torch.max(max_val, self.max_val))
            scale, zero_point = self._calculate_qparams(self.min_val, self.max_val)
            self.scale.copy_(scale)
            self.zero_point.copy_(zero_point)
        if self.fake_quant:
            return torch.fake_quantize_per_tensor_affine(x, self.scale, self.zero_point, self.q_min, self.q_max)
        else:
            return x


class PerChannelMinMaxObserver(Observer):
    def __init__(self, ch_axis=0, dtype: torch.dtype = torch.qint8,
                 qscheme: torch.qscheme = torch.per_channel_symmetric, **kwargs) -> None:
        super().__init__(dtype, qscheme, ch_axis, **kwargs)
        self.register_buffer("min_val", torch.tensor([]))
        self.register_buffer("max_val", torch.tensor([]))
        self.register_buffer("scale", torch.tensor([]))
        self.register_buffer("zero_point", torch.tensor([], dtype=torch.int32))

    def forward(self, x):
        if self.observe:
            if x.numel() == 0:
                return x
            x = x.detach()  # avoid keeping autograd tape
            min_val = self.min_val
            max_val = self.max_val
            x_dim = x.size()

            new_axis_list = list(range(len(x_dim)))
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            y = x.permute(new_axis_list)
            # Need to match dtype of min/max because the updates to buffers
            # are done in place and types need to match for comparisons
            y = y.to(self.min_val.dtype)
            y = torch.flatten(y, start_dim=1)
            if min_val.numel() == 0 or max_val.numel() == 0:
                min_val, max_val = torch.aminmax(y, dim=1)
            else:
                min_val_cur, max_val_cur = torch.aminmax(y, dim=1)
                min_val = torch.min(min_val_cur, min_val)
                max_val = torch.max(max_val_cur, max_val)
            self.min_val.resize_(min_val.shape)
            self.max_val.resize_(max_val.shape)
            self.min_val.copy_(min_val)
            self.max_val.copy_(max_val)
            scale, zero_point = self._calculate_qparams(self.min_val, self.max_val)
            self.scale.resize_(scale.shape)
            self.zero_point.resize_(zero_point.shape)
            self.scale.copy_(scale)
            self.zero_point.copy_(zero_point)
        if self.fake_quant:
            return torch.fake_quantize_per_channel_affine(x, self.scale, self.zero_point, self.ch_axis, self.q_min, self.q_max)
        else:
            return x


class BiasObserver(Observer):
    def __init__(self, w_ob: Observer, act_ob: Observer, **kwargs) -> None:
        dtype = torch.qint32
        is_w_per_channel = is_per_channel(w_ob.qscheme)
        qscheme = torch.per_channel_symmetric if is_w_per_channel else torch.per_tensor_symmetric
        super().__init__(dtype, qscheme, w_ob.ch_axis, **kwargs)
        self.w_ob = w_ob
        self.act_ob = act_ob

        self.register_buffer("scale", torch.tensor(1.))
        self.register_buffer("zero_point", torch.tensor(0, dtype=torch.int32))

    def forward(self, x):
        if self.observe:
            scale = self.w_ob.scale * self.act_ob.scale
            if self.per_channel:
                self.scale.data = torch.ones_like(scale)
                self.zero_point.data = torch.zeros_like(scale)
            self.scale.copy_(scale)
        if self.fake_quant:
            if self.per_channel:
                return torch.fake_quantize_per_channel_affine(
                    x, self.scale, self.zero_point, self.ch_axis, self.q_min, self.q_max
                )
            else:
                return torch.fake_quantize_per_tensor_affine(x, self.scale, self.zero_point, self.q_min, self.q_max)
        else:
            return x

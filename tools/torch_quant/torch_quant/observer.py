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
from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)

# definition of zero_point and scale:
# float_value = (quant_value - zero_point) * scale

# TODO: there is so many code patches to handle the scale & zero_point's shape.
# Need to try to see if there is a more efficient way.


class QParams(NamedTuple):
    qscheme: torch.qscheme
    dtype: torch.dtype
    scale: torch.Tensor
    zero_point: torch.Tensor
    ch_axis: int = -1  # -1 corresponding to per-tensor


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


def is_per_tensor(qscheme: torch.qscheme):
    return qscheme in (torch.per_tensor_symmetric, torch.per_tensor_affine)


def is_per_channel(qscheme: torch.qscheme):
    return qscheme in (torch.per_channel_symmetric, torch.per_channel_affine)


def is_symmetric(qscheme: torch.qscheme):
    return qscheme in (torch.per_tensor_symmetric, torch.per_channel_symmetric)


class Observer(torch.nn.Module, ABC):
    def __init__(self, dtype: torch.dtype, qscheme: torch.qscheme, ch_axis: int = -1, **kwargs) -> None:
        super().__init__()
        self.dtype = dtype
        self.qscheme = qscheme
        self.bit, self.signed = DTYPE_TO_BIT_SIGN[dtype]
        if self.symmetric and not self.signed:
            raise ValueError('Symmetric quantization requires signed dtype.')
        self.q_min, self.q_max = calc_quant_min_max(self.bit, self.signed)
        self.register_buffer('eps', torch.tensor(torch.finfo(torch.float32).eps))
        # TODO: to support dp & ddp, we should use tensor for observe and fake_quant.
        self.observe = True
        self.fake_quant = True
        self.ch_axis = ch_axis
        self._register_load_state_dict_pre_hook(partial(pre_load_state_dict_hook, self))

    @property
    def symmetric(self) -> bool:
        return is_symmetric(self.qscheme)

    @property
    def per_channel(self) -> bool:
        return is_per_channel(self.qscheme)

    @property
    def qparams(self) -> QParams:
        return QParams(qscheme=self.qscheme, dtype=self.dtype,
                       scale=self.scale,
                       zero_point=self.zero_point,
                       ch_axis=self.ch_axis)

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

        """
        LOGGER.debug(
            f'calc qparams: {self.min_val=}, {self.max_val=}, {self.q_min=}, {self.q_max=}, {self.bit=}, {self.signed=}, {scale=}, {zero_point=}')
        """
        return scale, zero_point

    @classmethod
    def from_qparams(cls, qparams: QParams):
        raise RuntimeError(f"Instantiating a {type(cls)} from QParams is not implemented")

    def set_mode(self, observe: bool, fake_quant: bool) -> None:
        self.observe = observe
        self.fake_quant = fake_quant


def toggle_observer(root: nn.Module, *, observe: bool, fake_quant: bool) -> None:
    for m in root.modules():
        if isinstance(m, Observer):
            m.set_mode(observe=observe, fake_quant=fake_quant)


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
        self.register_buffer("scale", torch.tensor(1.))
        self.register_buffer("zero_point", torch.tensor(0, dtype=torch.int32))

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
        if w_ob.per_channel or act_ob.per_channel:
            qscheme = torch.per_channel_symmetric
        else:
            qscheme = torch.per_tensor_symmetric
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


class FakeQuantizeLearnablePerChannelAffine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, ch_axis, quant_min, quant_max, grad_scale_factor):
        return torch._fake_quantize_learnable_per_channel_affine(
            x, scale, zero_point, ch_axis, quant_min, quant_max, grad_scale_factor
        )

    @staticmethod
    def symbolic(g, x, scale, zero_point, ch_axis, quant_min, quant_max, grad_scale_factor):
        # This is for torchscript generating by tracing.
        return g.op(
            "::FakeQuantizeLearnablePerChannelAffine",
            x, scale, zero_point,
            ch_axis_i=ch_axis, quant_min_i=quant_min, quant_max_i=quant_max
        )


class FakeQuantizeLearnablePerTensorAffine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, quant_min, quant_max, grad_scale_factor):
        return torch._fake_quantize_learnable_per_tensor_affine(
            x, scale, zero_point, quant_min, quant_max, grad_scale_factor
        )

    @staticmethod
    def symbolic(g, x, scale, zero_point, quant_min, quant_max, grad_scale_factor):
        # This is for torchscript generating by tracing.
        return g.op(
            "::FakeQuantizeLearnablePerTensorAffine",
            x, scale, zero_point, quant_min_i=quant_min, quant_max_i=quant_max
        )


class LSQObserver(Observer):
    def __init__(self,
                 ch_axis=0,
                 dtype: torch.dtype = torch.qint8,
                 qscheme: torch.qscheme = torch.per_tensor_symmetric,
                 init_scale: Optional[torch.tensor] = None,
                 init_zp: Optional[torch.tensor] = None,
                 **kwargs) -> None:
        super().__init__(dtype, qscheme, ch_axis, **kwargs)
        self.register_buffer('eps', torch.tensor([torch.finfo(torch.float32).eps]))
        self.scale = nn.Parameter(torch.tensor([1.]))
        self.zero_point = nn.Parameter(torch.tensor([0.]))

        if init_scale is not None:
            if self.per_channel:
                self.scale.data = torch.ones_like(init_scale)
            self.scale.data.copy_(init_scale)
        if init_zp is not None:
            if self.per_channel:
                self.zero_point.data = torch.zeros_like(init_zp).float()
            self.zero_point.data.copy_(init_zp)

    def forward(self, x):
        if self.fake_quant:
            if self.symmetric:
                self.zero_point.data.zero_()
            else:
                self.zero_point.data.clamp_(self.q_min, self.q_max).float()

            # TODO(bohua.cbh): figure out if per-channel grad_scale_factor
            # is beneficial.
            grad_scale_factor = 1.0 / ((self.q_max * x.numel()) ** 0.5)

            # torch.autograd.Function can be traced, but the torchscript cannot be
            # serialized locally. Unless, it is registered as a torch custom class
            # and the forward/serialize/deserialize functions are implemented. So we
            # implement three modes here to support different situations:
            # 1. "not torch.jit.is_tracing()" means the model are in ptq/qat mode.
            # 2. "torch.onnx.is_in_onnx_export()" means we are exporting the model to onnx
            # under the format required by some backends.
            # 3. Otherwise, we are exporting the model to torchscript with fake quant and
            # convert it to quantized torchscript using Blade.
            # Note: make sure that is_in_onnx_export() has a higher priority.
            if self.per_channel:
                if not torch.jit.is_tracing():
                    x = torch._fake_quantize_learnable_per_channel_affine(
                        x, self.scale, self.zero_point, self.ch_axis,
                        self.q_min, self.q_max, grad_scale_factor)
                elif torch.onnx.is_in_onnx_export():
                    x = FakeQuantizeLearnablePerChannelAffine.apply(
                        x, self.scale, self.zero_point, self.ch_axis,
                        self.q_min, self.q_max, grad_scale_factor
                    )
                else:
                    # for torch versions earlier than 1.10, we should use torch.long
                    x = torch.fake_quantize_per_channel_affine(
                        x, self.scale.data, self.zero_point.data.to(torch.int32),
                        self.ch_axis, self.q_min, self.q_max
                    )
            else:
                if not torch.jit.is_tracing():
                    x = torch._fake_quantize_learnable_per_tensor_affine(
                        x, self.scale, self.zero_point,
                        self.q_min, self.q_max, grad_scale_factor)
                elif torch.onnx.is_in_onnx_export():
                    x = FakeQuantizeLearnablePerTensorAffine.apply(
                        x, self.scale, self.zero_point,
                        self.q_min, self.q_max, grad_scale_factor)
                else:
                    x = torch.fake_quantize_per_tensor_affine(
                        x, float(self.scale), int(self.zero_point),
                        self.q_min, self.q_max
                    )
        return x

    @classmethod
    def from_qparams(cls, qparams: QParams):
        return cls(
            ch_axis=qparams.ch_axis,
            dtype=qparams.dtype,
            qscheme=qparams.qscheme,
            init_scale=qparams.scale,
            init_zp=qparams.zero_point)


# PyTorch's HistogramObserver with some code modification to
# satisfy out use requirement.
# https://github.com/pytorch/pytorch/blob/1ab112cfab5e9e5b3ec2521f0b4e6b93b6ff90d9/torch/ao/quantization/observer.py#L888
class HistogramObserver(Observer):
    def __init__(
        self,
        bins: int = 2048,
        upsample_rate: int = 128,
        dtype: torch.dtype = torch.quint8,
        qscheme=torch.per_tensor_affine
    ):
        if not is_per_tensor(qscheme):
            raise NotImplementedError(
                "HistogramObserver's qscheme only support torch.per_tensor_symmetric \
                    and torch.per_tensor_affine."
            )
        super().__init__(dtype, qscheme)
        self.bins = bins
        self.register_buffer("histogram", torch.zeros(self.bins))
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.register_buffer("scale", torch.tensor(1.))
        self.register_buffer("zero_point", torch.tensor(0, dtype=torch.int32))
        self.dst_nbins = 2 ** self.bit
        self.upsample_rate = upsample_rate

    def _get_norm(
            self, delta_begin: torch.Tensor, delta_end: torch.Tensor, density: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Compute the norm of the values uniformaly distributed between
        delta_begin and delta_end.
        Currently only L2 norm is supported.
        norm = density * (integral_{begin, end} x^2)
             = density * (end^3 - begin^3) / 3
        """
        norm = (
            delta_end * delta_end * delta_end - delta_begin * delta_begin * delta_begin
        ) / 3
        return density * norm

    def _compute_quantization_error(self, next_start_bin: int, next_end_bin: int):
        r"""
        Compute the quantization error if we use start_bin to end_bin as the
        min and max to do the quantization.
        """
        bin_width = (self.max_val.item() - self.min_val.item()) / self.bins

        dst_bin_width = bin_width * (next_end_bin - next_start_bin + 1) / self.dst_nbins
        if dst_bin_width == 0.0:
            return 0.0

        src_bin = torch.arange(self.bins, device=self.histogram.device)
        # distances from the beginning of first dst_bin to the beginning and
        # end of src_bin
        src_bin_begin = (src_bin - next_start_bin) * bin_width
        src_bin_end = src_bin_begin + bin_width

        # which dst_bins the beginning and end of src_bin belong to?
        dst_bin_of_begin = torch.clamp(
            torch.div(src_bin_begin, dst_bin_width, rounding_mode='floor'), 0, self.dst_nbins - 1
        )
        dst_bin_of_begin_center = (dst_bin_of_begin + 0.5) * dst_bin_width

        dst_bin_of_end = torch.clamp(
            torch.div(src_bin_end, dst_bin_width, rounding_mode='floor'), 0, self.dst_nbins - 1
        )
        dst_bin_of_end_center = (dst_bin_of_end + 0.5) * dst_bin_width

        density = self.histogram / bin_width

        norm = torch.zeros(self.bins, device=self.histogram.device)

        delta_begin = src_bin_begin - dst_bin_of_begin_center
        delta_end = dst_bin_width / 2
        norm += self._get_norm(delta_begin,
                               torch.ones(self.bins, device=self.histogram.device) * delta_end,
                               density)

        norm += (dst_bin_of_end - dst_bin_of_begin - 1) * self._get_norm(
            torch.tensor(-dst_bin_width / 2), torch.tensor(dst_bin_width / 2), density
        )

        dst_bin_of_end_center = dst_bin_of_end * dst_bin_width + dst_bin_width / 2

        delta_begin = -dst_bin_width / 2
        delta_end = src_bin_end - dst_bin_of_end_center
        norm += self._get_norm(torch.tensor(delta_begin), delta_end, density)

        return norm.sum().item()

    def _non_linear_param_search(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Non-linear parameter search.
        An approximation for L2 error minimization for selecting min/max.
        By selecting new min/max, we filter out outliers in input distribution.
        This follows the implementation of NormMinimization::NonlinearQuantizationParamsSearch in
        caffe2/quantization/server/norm_minimization.cc
        """
        assert self.histogram.size()[0] == self.bins, "bins mismatch"
        bin_width = (self.max_val - self.min_val) / self.bins

        # cumulative sum
        total = torch.sum(self.histogram).item()
        cSum = torch.cumsum(self.histogram, dim=0)

        stepsize = 1e-5  # granularity
        alpha = 0.0  # lower bound
        beta = 1.0  # upper bound
        start_bin = 0
        end_bin = self.bins - 1
        norm_min = float("inf")

        while alpha < beta:
            # Find the next step
            next_alpha = alpha + stepsize
            next_beta = beta - stepsize

            # find the left and right bins between the quantile bounds
            l = start_bin
            r = end_bin
            while l < end_bin and cSum[l] < next_alpha * total:
                l = l + 1
            while r > start_bin and cSum[r] > next_beta * total:
                r = r - 1

            # decide the next move
            next_start_bin = start_bin
            next_end_bin = end_bin
            if (l - start_bin) > (end_bin - r):
                # move the start bin
                next_start_bin = l
                alpha = next_alpha
            else:
                # move the end bin
                next_end_bin = r
                beta = next_beta

            if next_start_bin == start_bin and next_end_bin == end_bin:
                continue

            # calculate the quantization error using next_start_bin and next_end_bin
            norm = self._compute_quantization_error(next_start_bin, next_end_bin)

            if norm > norm_min:
                break
            norm_min = norm
            start_bin = next_start_bin
            end_bin = next_end_bin

        new_min = self.min_val + bin_width * start_bin
        new_max = self.min_val + bin_width * (end_bin + 1)
        return new_min, new_max

    def _adjust_min_max(
            self, combined_min: torch.Tensor, combined_max: torch.Tensor, upsample_rate: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        # We ensure that:
        # (combined_max - combined_min)/(downsample_rate*Nbins) = (max - min)/(upsample_rate*Nbins)
        # This allows us to have a common grid of resolution s, where we can align
        # the input histogram
        # start_idx maps min_val to the histogram bin index.

        # Compute the width of histogram bins is a straightforward solution, where
        # hist_bin_width = (self.max_val - self.min_val) / (self.bins * upsample_rate)
        # Underflow happens if the numerator is close to the smallest positive subnormal number of FP32
        # Therefore, we avoid such division operation.
        downsample_rate = int(
            torch.ceil(
                (combined_max - combined_min) * upsample_rate / (self.max_val - self.min_val)
            ).item()
        )
        e = downsample_rate * (self.max_val - self.min_val) / upsample_rate - (combined_max - combined_min)
        start_idx = int(
            torch.round(
                (self.min_val - combined_min) * self.bins * upsample_rate / (self.max_val - self.min_val)).item()
        )
        combined_max = combined_max + e
        combined_min = combined_min
        return combined_min, combined_max, downsample_rate, start_idx

    def _combine_histograms(
            self,
            orig_hist: torch.Tensor,
            new_hist: torch.Tensor,
            upsample_rate: int,
            downsample_rate: int,
            start_idx: int,
            Nbins: int,
    ) -> torch.Tensor:
        # First up-sample the histogram with new data by a factor of L
        # This creates an approximate probability density thats piecwise constant
        upsampled_histogram = new_hist.repeat_interleave(upsample_rate)
        # Now insert the upsampled histogram into the output
        # histogram, which is initialized with zeros.
        # The offset at which the histogram is introduced is determined
        # by the start index as the output histogram can cover a wider range
        histogram_with_output_range = torch.zeros(
            (Nbins * downsample_rate), device=orig_hist.device
        )
        histogram_with_output_range[
            start_idx: Nbins * upsample_rate + start_idx
        ] = upsampled_histogram
        # Compute integral histogram, double precision is needed to ensure
        # that there are no overflows
        integral_histogram = torch.cumsum(
            histogram_with_output_range, 0, dtype=torch.double
        )[downsample_rate - 1:: downsample_rate]
        # Finally perform interpolation
        shifted_integral_histogram = torch.zeros((Nbins), device=orig_hist.device)
        shifted_integral_histogram[1:Nbins] = integral_histogram[0:-1]
        interpolated_histogram = (
            integral_histogram - shifted_integral_histogram
        ) / upsample_rate
        orig_hist = orig_hist + interpolated_histogram.to(torch.float)
        return orig_hist

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        if self.observe:
            if x_orig.numel() == 0:
                return x_orig
            x = x_orig.detach()
            min_val = self.min_val
            max_val = self.max_val
            same_values = min_val.item() == max_val.item()
            is_uninitialized = min_val == float("inf") and max_val == float("-inf")
            if is_uninitialized or same_values:
                min_val, max_val = torch.aminmax(x)
                self.min_val.resize_(min_val.shape)
                self.min_val.copy_(min_val)
                self.max_val.resize_(max_val.shape)
                self.max_val.copy_(max_val)
                assert (
                    min_val.numel() == 1 and max_val.numel() == 1
                ), "histogram min/max values must be scalar."
                torch.histc(
                    x, self.bins, min=min_val, max=max_val, out=self.histogram  # type: ignore[arg-type]
                )
            else:
                new_min, new_max = torch.aminmax(x)
                combined_min = torch.min(new_min, min_val)
                combined_max = torch.max(new_max, max_val)
                # combine the existing histogram and new histogram into 1 histogram
                # We do this by first upsampling the histogram to a dense grid
                # and then downsampling the histogram efficiently
                (
                    combined_min,
                    combined_max,
                    downsample_rate,
                    start_idx,
                ) = self._adjust_min_max(combined_min, combined_max, self.upsample_rate)
                assert (
                    combined_min.numel() == 1 and combined_max.numel() == 1
                ), "histogram min/max values must be scalar."
                combined_histogram = torch.histc(
                    x, self.bins, min=combined_min, max=combined_max  # type: ignore[arg-type]
                )
                if combined_min == min_val and combined_max == max_val:
                    combined_histogram += self.histogram
                else:
                    combined_histogram = self._combine_histograms(
                        combined_histogram,
                        self.histogram,
                        self.upsample_rate,
                        downsample_rate,
                        start_idx,
                        self.bins,
                    )

                self.histogram.detach_().resize_(combined_histogram.shape)
                self.histogram.copy_(combined_histogram)
                self.min_val.detach_().resize_(combined_min.shape)
                self.min_val.copy_(combined_min)
                self.max_val.detach_().resize_(combined_max.shape)
                self.max_val.copy_(combined_max)
            scale, zero_point = self.calculate_qparams()
            self.scale.resize_(scale.shape)
            self.zero_point.resize_(zero_point.shape)
            self.scale.copy_(scale)
            self.zero_point.copy_(zero_point)

        if self.fake_quant:
            x_orig = torch.fake_quantize_per_tensor_affine(x_orig, self.scale, self.zero_point, self.q_min, self.q_max)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        is_uninitialized = self.min_val == float("inf") and self.max_val == float(
            "-inf"
        )
        if is_uninitialized:
            LOGGER.warning(
                "must run observer before calling calculate_qparams.\
                                    Returning default scale and zero point "
            )
            return torch.tensor([1.0], device=self.min_val.device.type), torch.tensor([0],
                                                                                      device=self.min_val.device.type)
        assert self.bins == len(self.histogram), (
            "The number of bins in histogram should be equal to the number of bins "
            "supplied while making this observer"
        )

        new_min, new_max = self._non_linear_param_search()

        return self._calculate_qparams(new_min, new_max)

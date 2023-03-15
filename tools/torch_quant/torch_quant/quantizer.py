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

from enum import Enum
from functools import partial
from typing import Callable, Dict, NamedTuple, Optional

import torch
import torch.nn as nn
from torch.fx import GraphModule, Tracer
from torch_quant.amp_module import get_fallback_names
from torch_quant.graph import (
    GraphModContext,
    fold_qdq,
    fuse_modules,
    insert_act_observer,
    observer_to_qdq,
    q_ref_dq_to_fbgemm,
    quantizable_module_to_amp,
    quantizable_module_to_observed,
    quantizable_module_to_ref,
    set_qconfig
)
from torch_quant.module import ModuleFilter, copy_and_replace, fx_trace
from torch_quant.observer import (
    BiasObserver,
    HistogramObserver,
    LSQObserver,
    MinMaxObserver,
    Observer,
    PerChannelMinMaxObserver,
    toggle_observer
)


class Backend(Enum):
    REFERENCE = 0
    DISC = 1
    FBGEMM = 2


DEFAULT_ACT_OB_CTR: Dict[Backend, Callable[..., Observer]] = {
    Backend.REFERENCE: partial(MinMaxObserver, dtype=torch.quint8, qscheme=torch.per_tensor_affine),
    Backend.DISC: partial(MinMaxObserver, dtype=torch.qint8, qscheme=torch.per_tensor_symmetric),
    Backend.FBGEMM: partial(HistogramObserver, dtype=torch.quint8, qscheme=torch.per_tensor_affine),
}

DEFAULT_W_OB_CTR = {
    Backend.REFERENCE: partial(MinMaxObserver, dtype=torch.quint8, qscheme=torch.per_tensor_affine),
    Backend.DISC: partial(PerChannelMinMaxObserver, dtype=torch.qint8, qscheme=torch.per_channel_symmetric),
    Backend.FBGEMM: partial(PerChannelMinMaxObserver, dtype=torch.qint8, qscheme=torch.per_channel_symmetric),
}

DEFAULT_BIAS_OB_CTR = BiasObserver

DEFAULT_QAT_W_OB_CTR = partial(LSQObserver, dtype=torch.qint8)
DEFAULT_QAT_ACT_OB_CTR: Dict[Backend, Callable[..., Observer]] = {
    Backend.REFERENCE: partial(LSQObserver, dtype=torch.quint8, qscheme=torch.per_tensor_affine),
    Backend.DISC: partial(LSQObserver, dtype=torch.qint8, qscheme=torch.per_tensor_symmetric),
    Backend.FBGEMM: partial(LSQObserver, dtype=torch.quint8, qscheme=torch.per_tensor_affine),
}


class ObserverTypes(NamedTuple):
    act_ob_ctr: Callable[..., Observer]
    w_ob_ctr: Callable[..., Observer]
    bias_ob_ctr: Optional[Callable[..., Observer]] = None

def get_observer_types(
        act_ob_ctr: Optional[Callable[..., Observer]],
        w_ob_ctr: Optional[Callable[..., Observer]],
        bias_ob_ctr: Optional[Callable[..., Observer]],
        default_act_ob_ctr: Callable[..., Observer],
        default_w_ob_ctr: Callable[..., Observer],
        default_bias_ob_ctr: Optional[Callable[..., Observer]]):
    act_ob_ctr = act_ob_ctr or default_act_ob_ctr
    w_ob_ctr = w_ob_ctr or default_w_ob_ctr
    bias_ob_ctr = bias_ob_ctr or default_bias_ob_ctr
    return ObserverTypes(act_ob_ctr=act_ob_ctr, w_ob_ctr=w_ob_ctr, bias_ob_ctr=bias_ob_ctr)


class Quantizer:
    def __init__(self, module_filter: Optional[ModuleFilter] = None,
                 backend: Backend = Backend.REFERENCE,
                 tracer: Optional[Tracer] = None,
                 ) -> None:
        self.module_filter = module_filter
        self.backend = backend
        self.tracer = tracer
        if backend == Backend.FBGEMM and torch.backends.quantized.engine != 'fbgemm':
            raise ValueError('fbgemm is not available, it only for x86_64')

    def calib_gm(
        self, name: str, gm: GraphModule, root: nn.Module, ob_types: ObserverTypes
    ) -> None:
        mf = self.module_filter.submodule_filter(name) if self.module_filter else None
        ctx = GraphModContext(
            gm, root, mf, ob_types.act_ob_ctr, ob_types.w_ob_ctr, ob_types.bias_ob_ctr
        )
        # TODO(litan.ls): unify graph modification for different backends
        if self.backend == Backend.DISC:
            ctx.modify_graph([
                set_qconfig,
                insert_act_observer,
                quantizable_module_to_observed,
            ])
        else:
            ctx.modify_graph([set_qconfig, fuse_modules, insert_act_observer])
        toggle_observer(gm, observe=True, fake_quant=False)

    def calib(self, model: nn.Module,
              act_ob_ctr: Optional[Callable[..., Observer]] = None,
              w_ob_ctr: Optional[Callable[..., Observer]] = None,
              bias_ob_ctr: Optional[Callable[..., Observer]] = None,
              ) -> nn.Module:
        ob_types = get_observer_types(
            act_ob_ctr, w_ob_ctr, bias_ob_ctr,
            DEFAULT_ACT_OB_CTR[self.backend], DEFAULT_W_OB_CTR[self.backend],
            DEFAULT_BIAS_OB_CTR)
        trace_mapping = fx_trace(model, self.module_filter, tracer=self.tracer)
        for name, traced in trace_mapping.items():
            self.calib_gm(name, traced.gm, traced.m, ob_types)
        return copy_and_replace(model, trace_mapping)

    def amp_gm(
        self, name: str, gm: GraphModule, root: nn.Module, ob_types: ObserverTypes
    ) -> None:
        mf = self.module_filter.submodule_filter(name) if self.module_filter else None
        ctx = GraphModContext(
            gm, root, mf, ob_types.act_ob_ctr, ob_types.w_ob_ctr, ob_types.bias_ob_ctr
        )
        if self.backend == Backend.DISC:
            ctx.modify_graph([set_qconfig, quantizable_module_to_amp])
        else:
            ctx.modify_graph([set_qconfig, fuse_modules, quantizable_module_to_amp])
        toggle_observer(gm, observe=False, fake_quant=True)

    def amp(
        self,
        model: nn.Module,
        act_ob_ctr: Optional[Callable[..., Observer]] = None,
        w_ob_ctr: Optional[Callable[..., Observer]] = None,
        bias_ob_ctr: Optional[Callable[..., Observer]] = None,
    ) -> nn.Module:
        ob_types = get_observer_types(
            act_ob_ctr,
            w_ob_ctr,
            bias_ob_ctr,
            DEFAULT_ACT_OB_CTR[self.backend],
            DEFAULT_W_OB_CTR[self.backend],
            DEFAULT_BIAS_OB_CTR,
        )
        trace_mapping = fx_trace(model, self.module_filter, tracer=self.tracer)
        for name, traced in trace_mapping.items():
            self.amp_gm(name, traced.gm, traced.m, ob_types)
        return copy_and_replace(model, trace_mapping)

    def fallback(self, model: nn.Module, num: int) -> None:
        self.module_filter = self.module_filter or ModuleFilter()
        self.module_filter.exclude_names = self.module_filter.exclude_names or list()
        self.module_filter.exclude_names.extend(get_fallback_names(model, num))

    def qat_gm(
        self, name: str, gm: GraphModule, root: nn.Module, ob_types: ObserverTypes
    ) -> None:
        mf = self.module_filter.submodule_filter(name) if self.module_filter else None
        ctx = GraphModContext(
            gm, root, mf, ob_types.act_ob_ctr, ob_types.w_ob_ctr, ob_types.bias_ob_ctr
        )
        ctx.modify_graph([
            set_qconfig,
            insert_act_observer,
            # Generally we do not add fake-quant to bias during qat fine-tuning. If
            # users want to evaluate the accuracy of a model in specific state, they
            # should use the model returned by `quantizer.quantize`
            quantizable_module_to_observed])
        toggle_observer(gm, observe=False, fake_quant=True)

    def qat(self, model: nn.Module,
            act_ob_ctr: Optional[Callable[..., Observer]] = None,
            w_ob_ctr: Optional[Callable[..., Observer]] = None,
            bias_ob_ctr: Optional[Callable[..., Observer]] = None,
            ) -> nn.Module:
        ob_types = get_observer_types(
            act_ob_ctr, w_ob_ctr, bias_ob_ctr,
            DEFAULT_QAT_ACT_OB_CTR[self.backend], DEFAULT_QAT_W_OB_CTR,
            None)
        trace_mapping = fx_trace(model, self.module_filter, tracer=self.tracer)
        for name, traced in trace_mapping.items():
            self.qat_gm(name, traced.gm, traced.m, ob_types)
        return copy_and_replace(model, trace_mapping)

    def quantize_gm(self, name: str, gm: GraphModule, root: nn.Module) -> None:
        mf = self.module_filter.submodule_filter(name) if self.module_filter else None
        ctx = GraphModContext(
            gm, root, mf, is_override_module=False, is_override_qconfig=False
        )
        if self.backend == Backend.DISC:
            ctx.modify_graph([
                set_qconfig,
                insert_act_observer,
                quantizable_module_to_observed,
            ])
            toggle_observer(gm, observe=False, fake_quant=True)

        elif self.backend == Backend.REFERENCE:
            ctx.modify_graph([
                set_qconfig,
                fuse_modules,
                insert_act_observer,
                observer_to_qdq,
                quantizable_module_to_ref,
            ])
        elif self.backend == Backend.FBGEMM:
            ctx.modify_graph([
                set_qconfig,
                fuse_modules,
                insert_act_observer,
                observer_to_qdq,
                quantizable_module_to_ref,
                q_ref_dq_to_fbgemm,
                fold_qdq,
            ])
        else:
            raise ValueError(f'Unsupported backend {self.backend.name}')
        # remove unused modules (e.g. observers) or the following tracing might fail
        ctx.gm.delete_all_unused_submodules()

    def quantize(self, model: nn.Module) -> nn.Module:
        trace_mapping = fx_trace(model, self.module_filter, tracer=self.tracer)
        for name, traced in trace_mapping.items():
            self.quantize_gm(name, traced.gm, traced.m)
        return copy_and_replace(model, trace_mapping)

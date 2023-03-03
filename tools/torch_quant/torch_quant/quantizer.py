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
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.fx import GraphModule, Tracer
from torch_quant.graph import (
    GraphModContext,
    fold_qdq,
    fuse_modules,
    insert_act_observer,
    observer_to_qdq,
    q_ref_dq_to_fbgemm,
    quantizable_module_to_observed,
    quantizable_module_to_ref,
    set_qconfig
)
from torch_quant.module import ModuleFilter, copy_and_replace, fx_trace
from torch_quant.observer import (
    BiasObserver,
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
    Backend.FBGEMM: partial(MinMaxObserver, dtype=torch.quint8, qscheme=torch.per_tensor_affine),
}

DEFAULT_W_OB_CTR = {
    # According to the url below, PyTorch's reference module does not support
    # symmetric quantization, which is confusing...
    # https://github.com/pytorch/pytorch/blob/28e69954a1fb25c20153c0e3636b9052e6962ffa/torch/ao/nn/quantized/reference/modules/utils.py#L19
    Backend.REFERENCE: partial(MinMaxObserver, dtype=torch.quint8, qscheme=torch.per_tensor_affine),
    Backend.DISC: partial(PerChannelMinMaxObserver, dtype=torch.qint8, qscheme=torch.per_channel_symmetric),
    Backend.FBGEMM: partial(PerChannelMinMaxObserver, dtype=torch.qint8, qscheme=torch.per_channel_symmetric),
}

DEFAULT_BIAS_OB_CTR = BiasObserver


class Quantizer:
    def __init__(self, module_filter: Optional[ModuleFilter] = None,
                 backend: Backend = Backend.REFERENCE,
                 tracer: Optional[Tracer] = None,
                 act_ob_ctr: Optional[Callable[..., Observer]] = None,
                 w_ob_ctr: Optional[Callable[..., Observer]] = None,
                 bias_ob_ctr: Optional[Callable[..., Observer]] = None,
                 ) -> None:
        self.module_filter = module_filter
        self.backend = backend
        self.tracer = tracer
        self.act_ob_ctr = act_ob_ctr or DEFAULT_ACT_OB_CTR[backend]
        self.w_ob_ctr = w_ob_ctr or DEFAULT_W_OB_CTR[backend]
        self.bias_ob_ctr = bias_ob_ctr or DEFAULT_BIAS_OB_CTR

    def calib_gm(self, gm: GraphModule, root: nn.Module, name: str) -> None:
        module_filter = self.module_filter or ModuleFilter()
        module_filter = module_filter.submodule_filter(name)
        ctx = GraphModContext(
            gm, root, self.act_ob_ctr, self.w_ob_ctr, self.bias_ob_ctr, module_filter
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

    def calib(self, model: nn.Module) -> nn.Module:
        trace_mapping = fx_trace(model, self.module_filter, tracer=self.tracer)
        for name, traced in trace_mapping.items():
            self.calib_gm(traced.gm, traced.m, name)
        return copy_and_replace(model, trace_mapping)

    def quantize_gm(self, gm: GraphModule, root: nn.Module, name: str) -> None:
        module_filter = self.module_filter or ModuleFilter()
        module_filter = module_filter.submodule_filter(name)
        ctx = GraphModContext(
            gm, root, self.act_ob_ctr, self.w_ob_ctr, self.bias_ob_ctr, module_filter
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
            self.quantize_gm(traced.gm, traced.m, name)
        return copy_and_replace(model, trace_mapping)

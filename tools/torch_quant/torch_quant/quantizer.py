from enum import Enum
from typing import Optional

import torch.nn as nn
from torch.fx import GraphModule, Tracer

from torch_quant.graph import (fold_qdq, insert_act_observer, modify_graph,
                               observer_to_qdq, q_ref_dq_to_fbgemm,
                               quantizable_module_to_ref, set_qconfig)
from torch_quant.module import ModuleFilter, copy_and_replace, fx_trace
from torch_quant.observer import toggle_observer


class Backend(Enum):
    REFERENCE = 0
    DISC = 1
    FBGEMM = 2


class Quantizer:
    def __init__(self, module_filter: Optional[ModuleFilter] = None,
                 backend: Optional[Backend] = Backend.REFERENCE,
                 tracer: Optional[Tracer] = None) -> None:
        self.module_filter = module_filter
        self.backend = backend
        self.tracer = tracer

    def calib_gm(self,  gm: GraphModule, root: nn.Module) -> None:
        modify_graph(gm, root, [set_qconfig, insert_act_observer])
        toggle_observer(gm, observe=True, fake_quant=False)

    def calib(self, model: nn.Module) -> nn.Module:
        trace_mapping = fx_trace(model, self.module_filter, tracer=self.tracer)
        for x in trace_mapping.values():
            self.calib_gm(x.gm, x.m)
        return copy_and_replace(model, trace_mapping)

    def quantize_gm(self, gm: GraphModule, root: nn.Module) -> None:
        passes = [set_qconfig, insert_act_observer,
                  observer_to_qdq, quantizable_module_to_ref]
        if self.backend == Backend.REFERENCE:
            ...
        elif self.backend == Backend.DISC:
            raise NotImplementedError(
                f'Backend {self.backend.name} has not been implemented yet.')
        elif self.backend == Backend.FBGEMM:
            passes += [q_ref_dq_to_fbgemm, fold_qdq]
        else:
            raise ValueError(f'Unsupported backend {self.backend.name}')

        modify_graph(gm, root, passes)

    def quantize(self, model: nn.Module) -> nn.Module:
        trace_mapping = fx_trace(model, self.module_filter, tracer=self.tracer)
        for x in trace_mapping.values():
            self.quantize_gm(x.gm, x.m)
        return copy_and_replace(model, trace_mapping)

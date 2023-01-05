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

import copy
from typing import Dict, List, NamedTuple, Optional, Type

import torch.nn as nn
from torch.fx import GraphModule, Tracer


class TracePair(NamedTuple):
    gm: GraphModule
    m: nn.Module


# eval priority: exclude > include, name > type
class ModuleFilter(NamedTuple):
    include_names: Optional[List[str]] = None
    include_types: Optional[List[Type[nn.Module]]] = None
    exclude_names: Optional[List[str]] = None
    exclude_types: Optional[List[Type[nn.Module]]] = None


def fx_trace(root: nn.Module, module_filter: Optional[ModuleFilter] = None,
             tracer: Optional[Tracer] = None) -> Dict[str, TracePair]:
    if tracer is None:
        tracer = Tracer()
    if module_filter is None:
        fx_graph = tracer.trace(root)
        return {'': TracePair(gm=GraphModule(root, fx_graph), m=root)}
    else:
        # TODO(litan.ls): customize tracer to apply filters
        raise NotImplementedError()


def copy_and_replace(root: nn.Module, trace_mapping: Dict[str, TracePair]) -> nn.Module:
    def _search(node: nn.Module, name: str):
        if name in trace_mapping:
            return trace_mapping[name].gm
        copied = copy.copy(node)
        for n, c in copied.named_children():
            # TODO(litan.ls): test correctness of module replacement.
            copied.add_module(n, _search(c, f'{name}.{n}'))
        return copied
    return _search(root, '')

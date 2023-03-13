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
class ModuleFilter:
    def __init__(
        self,
        include_names: Optional[List[str]] = None,
        include_classes: Optional[List[Type[nn.Module]]] = None,
        include_op_types: Optional[List[Type[nn.Module]]] = None,
        exclude_names: Optional[List[str]] = None,
        exclude_classes: Optional[List[Type[nn.Module]]] = None,
        exclude_op_types: Optional[List[Type[nn.Module]]] = None,
    ):
        """
        Args:
            include_names:
                List of modules to quantize, identified by name.
                If not None, only the modules in this list are quantized.
                Example: ['foo.bar.name']
            include_classes:
                List of modules to quantize, identified by class (custom module).
                If not None, only the modules in this list are quantized.
                Example: ['MyModule']
            include_op_types:
                Specify the types of operators to quantize.
                For [torch.nn.Conv2d], it will quantize torch.nn.Conv2d only.
                It quantizes all supported operators by default.
            exclude_names:
                List of modules to exclude, identified by name.
                If not None, modules in this list will be excluded from quantization.
                Example: ['foo.bar.name']
            exclude_classes:
                List of modules to exclude, identified by class (custom module).
                If not None, modules in this list will be excluded from quantization.
                Example: ['MyModule']
            exclude_op_types:
                Specify the types of operators not to be quantized.
                For [torch.nn.Conv2d], it will quantize all supported operators except
                torch.nn.Conv2d.
        """
        self.include_names = include_names
        self.include_classes = include_classes
        self.include_op_types = include_op_types
        self.exclude_names = exclude_names
        self.exclude_classes = exclude_classes
        self.exclude_op_types = exclude_op_types

    def _submodule_names(self, names, module_name: str) -> Optional[List[str]]:
        """
        If module name is 'foo', turn full path 'foo.bar.name' into 'bar.name'
        """
        if names and module_name:
            lstrip_func = lambda x : x.replace(f'{module_name}.', '', 1)
            names = [lstrip_func(m) for m in names if m.startswith(f'{module_name}.')]
        return names or None

    def submodule_filter(self, module_name: str):
        module_filter = ModuleFilter(
            include_names=self._submodule_names(self.include_names, module_name),
            include_classes=self.include_classes,
            include_op_types=self.include_op_types,
            exclude_names=self._submodule_names(self.exclude_names, module_name),
            exclude_classes=self.exclude_classes,
            exclude_op_types=self.exclude_op_types,
        )
        return module_filter


class PatchTracer:
    def __init__(
        self,
        tracer: Tracer,
        exclude_names: Optional[List[str]] = None,
        exclude_classes: Optional[List[Type[nn.Module]]] = None,
    ):
        self.tracer = tracer
        self.exclude_names = exclude_names
        self.exclude_classes = exclude_classes

    def __enter__(self):
        self.tracer.exclude_names = self.exclude_names
        self.tracer.exclude_classes = self.exclude_classes
        self.tracer._original_is_leaf_module = self.tracer.is_leaf_module

        def _patched_is_leaf_module(self, m, module_qualified_name):
            if self.exclude_names and module_qualified_name in self.exclude_names:
                return True
            if self.exclude_classes and type(m) in self.exclude_classes:
                return True
            return self._original_is_leaf_module(m, module_qualified_name)

        self.tracer.is_leaf_module = _patched_is_leaf_module.__get__(
            self.tracer, type(self.tracer)
        )
        return self.tracer

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tracer.is_leaf_module = self.tracer._original_is_leaf_module


def fx_trace(root: nn.Module, module_filter: Optional[ModuleFilter] = None,
             tracer: Optional[Tracer] = None) -> Dict[str, TracePair]:
    if tracer is None:
        tracer = Tracer()
    if module_filter is None:
        fx_graph = tracer.trace(root)
        return {'': TracePair(gm=GraphModule(root, fx_graph), m=root)}
    else:
        in_names, in_types = module_filter.include_names, module_filter.include_classes
        ex_names, ex_types = module_filter.exclude_names, module_filter.exclude_classes
        if in_names or in_types:
            trace_mapping: Dict[str, TracePair] = dict()
            for n, m in root.named_modules():
                if (in_names and n in in_names) or (in_types and type(m) in in_types):
                    module_ex_names = module_filter.submodule_filter(n).exclude_names
                    if module_ex_names or ex_types:
                        with PatchTracer(tracer, module_ex_names, ex_types) as tracer:
                            fx_graph = tracer.trace(copy.deepcopy(m))
                    else:
                        fx_graph = tracer.trace(copy.deepcopy(m))
                    trace_mapping[n] = TracePair(gm=GraphModule(m, fx_graph), m=m)
            return trace_mapping
        else:
            with PatchTracer(tracer, ex_names, ex_types) as tracer:
                fx_graph = tracer.trace(root)
            return {'': TracePair(gm=GraphModule(root, fx_graph), m=root)}


def copy_and_replace(root: nn.Module, trace_mapping: Dict[str, TracePair]) -> nn.Module:
    def _parent_name(name):
        """
        Turn 'foo.bar.name' into ['foo.bar', 'name']
        """
        r = name.rsplit('.', 1)
        if len(r) == 1:
            return '', r[0]
        else:
            return r[0], r[1]

    def _update_module(modules, target, new_module):
        parent_name, name = _parent_name(target)
        setattr(modules[parent_name], name, new_module)

    if '' in trace_mapping:
        return trace_mapping[''].gm
    copied = copy.deepcopy(root)
    root_modules = dict(root.named_modules())
    copied_modules = dict(copied.named_modules())
    for name, traced in trace_mapping.items():
        _update_module(root_modules, name, traced.m)
        _update_module(copied_modules, name, traced.gm)
    return copied

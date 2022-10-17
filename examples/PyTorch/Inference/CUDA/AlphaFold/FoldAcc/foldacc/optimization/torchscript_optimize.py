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
import types
import copy
import torch

from foldacc.optimization.utils import (
    register_forward_hook, 
    convert_dummy_inputs,
    wrap_model_inputs,
    patch_model_forward,
    print_logger,
    check_tensor
)

logger = logging.getLogger("foldacc")

def find_trace_inputs(func_varnames, chunk_varnames, inputs):
    new_inputs = []
    for i, var in enumerate(func_varnames[1:]):
        if i < len(inputs[0]):
            if chunk_varnames[i + 1] == var:
                new_inputs.append(inputs[0][i])
            else:
                raise ValueError(f"{var} != {chunk_varnames[i + 1]}.")
        else:
            for k, v in inputs[1].items():
                if var == k:
                    new_inputs.append(v)

    return new_inputs

def replace_chunk_func(model, func_name, varnames, scripted_module):
    model.chunk_scripted_module = scripted_module

    def patch_func(old_func):
        def new_func(module, *args, **kwargs):
            inputs = find_trace_inputs(varnames[0], varnames[1], [args, kwargs])
            return module.chunk_scripted_module(*inputs)
        return new_func

    model.__setattr__(func_name, types.MethodType(patch_func(model.__getattribute__(func_name)), model))
    return model

def convert_chunk_layer(module, inputs, config, device):
    chunk_func_name, trace_func, script_class = config

    trace_func_varnames = trace_func.__code__.co_varnames[
        :trace_func.__code__.co_argcount]
    chunk_func_varnames = module.__getattribute__(chunk_func_name).__code__.co_varnames[
        :module.__getattribute__(chunk_func_name).__code__.co_argcount]

    # get trace_func inputs
    func_inputs = []
    def patch_chunk_func(model, func_name):
        def patch_func(old_func):
            def new_func(_, *args, **kwargs):
                func_inputs.extend([args, kwargs])
                return old_func(*args, **kwargs)
            return new_func

        model.__setattr__(func_name, types.MethodType(patch_func(model.__getattribute__(func_name)), model))
        return model

    if hasattr(module, trace_func.__name__):
        module = patch_chunk_func(module, trace_func.__name__)
        trace_input_names = trace_func_varnames
    else:
        module = patch_chunk_func(module, chunk_func_name)
        trace_input_names = chunk_func_varnames

    org_output = module(*inputs[0], **inputs[1])

    if len(func_inputs) == 0:
        return module

    # trace func in while loop
    old_forward = module.forward
    module.forward = types.MethodType(trace_func, module)

    trace_varnames = trace_func.__code__.co_varnames[:trace_func.__code__.co_argcount]
    traced_inputs = find_trace_inputs(trace_varnames, trace_input_names, func_inputs)
    traced_inputs = convert_dummy_inputs(list(traced_inputs), device)
    traced_module = torch.jit.trace(module, traced_inputs, check_trace=False)

    # script while loop func
    script_module = script_class(traced_module)
    script_varnames = script_module.forward.__code__.co_varnames[:script_module.forward.__code__.co_argcount]
    scripted_module = torch.jit.script(script_module)

    # replace chunk_func by scripted module
    module = replace_chunk_func(module, chunk_func_name, (script_varnames, chunk_func_varnames), scripted_module)
    module.forward = old_forward

    # TODO: check output
    # optim_output = module(*inputs[0], **inputs[1])

    return module

def torchscript_optimize(model, inputs, optimize_config):
    """ convert model to torchscript model """
    if not optimize_config.enable_trace:
        return model

    org_model = copy.deepcopy(model)

    chunking_modules = optimize_config.chunking_modules

    module_inputs = {}
    def forward_hook(module, *args, **kwargs):
        module_inputs[module.name] = (args, kwargs)

    convert_modules = []
    module_old_forward = {}
    submodules = [(name, module) for name, module in model.named_modules()]
    for name, module in reversed(submodules):
        if type(module) in chunking_modules.keys():
            module.name = name
            convert_modules.append((name, module, chunking_modules[type(module)]))

            module, old_forward = register_forward_hook(module, forward_hook)
            module_old_forward[name] = old_forward
    
    # forward model to obtain inputs of convert modules
    inputs_clone = copy.deepcopy(inputs)
    org_output = model(*inputs_clone)

    # convert module has chunk layer to torchscript
    for name, module, config in convert_modules:
        if name not in module_inputs:
            print_logger(logger.warning, f"can not obtain {name} input")
            continue

        module_input = module_inputs[name]
        module = convert_chunk_layer(module, module_input, config, optimize_config.device)
        module.forward = module_old_forward[name]

        print_logger(logger.debug, f"convert chunking layer in {name} to torchscript")

    # convert model to torchscript
    inputs = convert_dummy_inputs(list(inputs), optimize_config.device)

    inputs_clone = copy.deepcopy(inputs)
    traced_model = torch.jit.trace(model, inputs_clone, check_trace=False)

    inputs_clone = copy.deepcopy(inputs)
    traced_output = traced_model(*inputs_clone)

    if not check_tensor(traced_output, org_output, optimize_config.trace_check_tolerance):
        print_logger(logger.info, f"torchscript results check failed")
        return org_model

    # convert input to torch.tensor
    model = wrap_model_inputs(traced_model)

    return model
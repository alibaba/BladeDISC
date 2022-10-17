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

import types
from typing import Tuple, List, Callable, Any, Dict, Sequence, Optional

import torch

def print_logger(log_func, text):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        log_func(text)

def patch_model_forward(model, before_tasks=[], after_task=[]):
    ''' patch forward func to run tasks'''

    def patch_forward(old_forward):
        def new_forward(_, *args, **kwargs):
            for task in before_tasks:
                args, kwargs = task(*args, **kwargs)

            outputs = old_forward(*args, **kwargs)

            for task in after_task:
                outputs = task(outputs)
            
            return outputs
        return new_forward

    model.forward = types.MethodType(patch_forward(model.forward), model)

    return model

def register_forward_hook(model, forward_hook):
    ''' register forward func'''

    def patch_forward(old_forward):
        def new_forward(module, *args, **kwargs):
            forward_hook(module, *args, **kwargs)
            return old_forward(*args, **kwargs)
        return new_forward

    old_forward = model.forward
    model.forward = types.MethodType(patch_forward(model.forward), model)

    return model, old_forward

def check_tensor(outputs, target, tolerance=1e-3):
    if isinstance(outputs, torch.Tensor):
        diff = torch.mean(1.0*(torch.abs(outputs - target) < tolerance))
        if diff < 1.0:
            return False
        return True
    elif type(outputs) in [list, tuple]:
        for i, out in enumerate(outputs):
            if not check_tensor(out, target[i], tolerance):
                return False
        return True
    
    return True

def convert_dummy_inputs(inputs, device):
    ''' convert inputs to torch.Tensor '''
    new_inputs = []
    for i, arg in enumerate(inputs):
        if isinstance(arg, torch.Tensor):
            new_inputs.append(arg)
        elif type(arg) in [list]:
            new_arg = convert_dummy_inputs(arg, device)
            new_inputs.append(list(new_arg))
        elif type(arg) in [tuple, list]:
            new_arg = convert_dummy_inputs(list(arg), device)
            new_inputs.append(new_arg)
        elif type(arg) in [dict]:
            new_arg = convert_dummy_inputs([v for k, v in arg.items()], device)
            new_inputs.append(new_arg)
        else:
            try:
                new_arg = torch.tensor(arg).to(device)
                new_inputs.append(new_arg)
            except:
                raise ValueError(f"cannot convert {arg} to torch.Tensor")
    
    return tuple(new_inputs)

def wrap_model_inputs(model, only_args=True, default_inputs=None):
    ''' wrap forward func to ignore some inputs '''

    def patch_forward(old_forward):
        def new_forward(_, *args, **kwargs):
            new_args = []
            for i, arg in enumerate(args):
                if type(arg) in [int]:
                    new_args.append(torch.tensor(arg).cuda())
                elif isinstance(arg, torch.Tensor):
                    new_args.append(arg)
            new_kwargs = {}
            for key, value in kwargs.items():
                if type(value) in [int]:
                    if only_args:
                        new_args.append(torch.tensor(value).cuda())
                    else:
                        new_kwargs[key] = torch.tensor(value).cuda()
                elif isinstance(value, torch.Tensor):
                    if only_args:
                        new_args.append(value)
                    else:
                        new_kwargs[key] = value
            kwargs = new_kwargs
            if default_inputs is not None:
                kwargs.update(default_inputs)
            args = tuple(new_args)

            return old_forward(*args, **kwargs)
        return new_forward

    model.forward = types.MethodType(patch_forward(model.forward), model)

    return model
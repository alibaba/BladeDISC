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
import copy
import torch

from foldacc.optimization.utils import (
    patch_model_forward, 
    register_forward_hook, 
    check_tensor,
    print_logger,
)

logger = logging.getLogger("foldacc")

def auto_mix_precision_optimize(model, inputs, optimize_config):
    enable_amp = optimize_config.enable_auto_mix_precision
    precision = optimize_config.precision
    
    # cast input tensor to low precision
    def input_cast(*args, **kwargs):
        new_args = []
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor) and arg.dtype != precision:
                new_args.append(arg.to(precision))
            else:
                new_args.append(arg)

        new_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and value.dtype != precision:
                new_kwargs[key] = value.to(precision)
            else:
                new_kwargs[key] = value

        return tuple(new_args), new_kwargs
    
    # cast output tensor to float32
    def output_cast(outputs):
        if isinstance(outputs, torch.Tensor):
            if outputs.dtype != model.dtype:
                outputs = outputs.to(model.dtype)
            
            return outputs

        new_outputs = []
        for out in outputs:
            if isinstance(out, torch.Tensor) and out.dtype != model.dtype:
                new_outputs.append(out.to(model.dtype))
            else:
                new_outputs.append(out)
        return tuple(new_outputs)
    
    # only do AMP for half
    if enable_amp and precision == torch.half and precision != model.dtype:
        # check wether use module
        use_module = {}
        def forward_hook(module, *args, **kwargs):
            use_module[module.name] = module
        
        low_precision_modules = []
        for name, module in model.named_modules():
            if type(module) in optimize_config.low_precision_modules and module not in low_precision_modules:
                module.name = name
                register_forward_hook(module, forward_hook)
                low_precision_modules.extend([m for n, m in module.named_modules()])

        dummy_input = copy.deepcopy(inputs)
        org_outputs = model(*dummy_input)

        for name, module in use_module.items():
            old_forward = module.forward
            patch_model_forward(module, [input_cast], [output_cast])
            module = module.to(precision)
            if hasattr(module, "inf"):
                old_inf = module.inf
                module.inf = 1e4
            if hasattr(module, "eps"):
                old_eps = module.eps
                module.eps = 1e-4
            print_logger(logger.debug, f"cast {name} to {precision}")

            dummy_input = copy.deepcopy(inputs)
            outputs = model(*dummy_input)
            
            if not check_tensor(outputs, org_outputs, optimize_config.check_tolerance):
                # fallback to float32
                module.forward = old_forward
                module = module.to(model.dtype)
                if hasattr(module, "inf"):
                    module.inf = old_inf
                if hasattr(module, "eps"):
                    module.eps = old_eps
                print_logger(logger.debug, f"fallback {name}")
                
    else:
        if precision != model.dtype:
            model = patch_model_forward(model, [input_cast], [output_cast])
            model = model.to(precision)

        print_logger(logger.debug, f"cast model to {precision}")
    return model
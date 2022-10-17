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

import os
import logging
import types
import sys
import onnx
import torch

from foldacc.optimization.utils import convert_dummy_inputs, wrap_model_inputs
from foldacc.optimization.distributed.convert import convert_save_model, convert_load_model

logger = logging.getLogger("foldacc")

try:
    import torch_blade
    enable_disc = True
except:
    logger.debug("Import BladeDisc Failed!")
    enable_disc = False

def bladedisc_optimize(model, inputs, optimize_config):
    if not optimize_config.enable_bladedisc or not optimize_config.enable_trace:
        return model

    if not isinstance(model, torch.jit.ScriptModule):
        return model

    if enable_disc and optimize_config.precision != torch.bfloat16:
        # replace pythonop by customop to avoid optimize fail
        model = convert_save_model(model)

        inputs = convert_dummy_inputs(list(inputs), optimize_config.device)
        
        other_tag = ""
        if torch.distributed.is_initialized():
            other_tag = f"_{torch.distributed.get_rank()}"
        
        save_path = f"{optimize_config.temp_dir}/save_model{other_tag}.pt"
        data_path = f"{optimize_config.temp_dir}/input_data{other_tag}.pth"
        load_path = f"{optimize_config.temp_dir}/load_model{other_tag}.pt"

        torch.jit.save(model, save_path)
        torch.save(inputs, data_path)
        if os.path.exists(load_path):
            os.remove(load_path)

        if isinstance(optimize_config.device, str):
            device = int(optimize_config.device.split(":")[-1])
        else:
            device = optimize_config.device

        os.system(f"{sys.executable} {os.path.dirname(__file__)}/run_bladedisc.py {save_path} {data_path} {load_path} {device} > /dev/null 2>&1")

        if os.path.exists(load_path):
            model = torch.jit.load(load_path)

        model = convert_load_model(model)
        model = wrap_model_inputs(model)

    return model
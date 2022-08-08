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

import torch
import torch.nn as nn
import logging
import copy
import os
import tempfile

from foldacc.optimization import (
    auto_mix_precision_optimize, 
    bladedisc_optimize,
    torchscript_optimize
)

logging.basicConfig(format='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger("foldacc").setLevel(logging.INFO)
logger = logging.getLogger("foldacc")

optimize_pipeline = [
    auto_mix_precision_optimize,
    torchscript_optimize,
    bladedisc_optimize,
]

class Config:
    def __init__(self):
        self.precision = torch.float
        
        # When self.enable_auto_mix_precision = True, 
        # we will set module in self.low_precision_modules to self.precision,
        # other modules will be set to float32.
        self.enable_auto_mix_precision = True
        self.enable_bladedisc = True
        self.enable_trace = True

        self.low_precision_modules = [nn.Linear]
        self.check_tolerance = 1e-3

        self.chunking_modules = {}

        self.device = "cuda:0"

        self.temp = tempfile.TemporaryDirectory()
        self.temp_dir = self.temp.name

    def __del__(self):
        self.temp.cleanup()
    
    def clean_tmp(self):
        self.temp.cleanup()
        self.temp = tempfile.TemporaryDirectory()
        self.temp_dir = self.temp.name

    def add_low_precision_module(self, module_type):
        if isinstance(module_type, list):
            self.low_precision_modules.extend(module_type)
        else:
            self.low_precision_modules.append(module_type)

    def add_chunking_module(self, module_type, chunk_func_name, trace_func, script_func):
        self.chunking_modules[module_type] = (chunk_func_name, trace_func, script_func)

def optimize(model, inputs, optimize_config):
    optim_model = copy.deepcopy(model)
    
    for optim in optimize_pipeline:
        try:
            optim_model = optim(optim_model, inputs, optimize_config)
        except:
            logger.warning(f"running [{optim.__name__}] failed.")

    return optim_model


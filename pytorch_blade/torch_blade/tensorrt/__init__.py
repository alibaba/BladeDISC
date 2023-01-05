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

try:
    from torch_blade._torch_blade._tensorrt import *
    from torch_blade.config import OptPipelines
    from torch_blade.tensorrt.flags import builder_flags_context
    from torch_blade.tensorrt.tensorrt_optimization import (
        _TRT_GROUP_NAME,
        get_unsupported_nodes,
        optimize_trt,
        trt_engine_conversion
    )

    OptPipelines.register_pipeline(backend_name(), optimize_trt)
    _TRT_NAME = backend_name()
    _is_available = True
except ImportError:
    _is_available = False
    _TRT_NAME = "TensorRT"

from torch_blade import utils


def is_available():
    return _is_available


def collect_engines(script_module):
    """
    Collect all engines in a script_module of the group type
    """
    return utils.collect_engines(script_module, _TRT_GROUP_NAME)


def num_engines(script_module):
    """
    Return the number of engines of the group_type
    """
    return utils.num_engines(script_module, _TRT_GROUP_NAME)


def num_compiled_nodes(script_module):
    """
    Return the number of nodes compiled by TRT
    """
    return utils.num_compiled_nodes(script_module, _TRT_GROUP_NAME)


def _fetch_trt_tensors(script_module):
    last_inputs = dict()
    last_outputs = dict()
    trt_engines = collect_engines(script_module)
    for trt_name, trt_grp in trt_engines:
        last_inputs[trt_name] = trt_grp.last_inputs()
        last_outputs[trt_name] = trt_grp.last_outputs()
    return last_inputs, last_outputs

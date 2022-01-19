# Copyright 2021 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import torch
from .._torch_blade._tools import *
from .._torch_blade._tools import _jit_pass_onnx
from .._torch_blade._tools import _jit_pass_lower_simple_tuples
from .._torch_blade._tools import _jit_pass_const_loop_unrolling


@contextlib.contextmanager
def trust_tracing_shape(flag=True):
    old_flag = set_trust_tracing_shape(flag)
    try:
        yield
    finally:
        set_trust_tracing_shape(old_flag)

@contextlib.contextmanager
def record_cluster_io_context(flag=True):
    old_flag = set_record_cluster_io_flag(flag)
    try:
        yield
    finally:
        set_record_cluster_io_flag(old_flag)

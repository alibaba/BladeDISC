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

try:
    from .._torch_blade._mlir import *
    _is_available = True
except ImportError as e:
    # MLIR support is disable
    _is_available = False

import contextlib
from torch_blade import utils
from torch_blade.config import OptPipelines
from torch_blade.mlir.disc_engine_conversion import (
    _optimize_mlir,
    _compile_torchscript,
)

_DISC_GROUP_NAME = "disc_grp"
_DISC_TESTING_CONTEXT = False
OptPipelines.register_pipeline("DISC", _optimize_mlir)

def is_available():
    return _is_available

@contextlib.contextmanager
def testing_context():
    global _DISC_TESTING_CONTEXT
    old_mlir_testing_context = _DISC_TESTING_CONTEXT
    try:
        _DISC_TESTING_CONTEXT = True
        yield
    finally:
        _DISC_TESTING_CONTEXT = old_mlir_testing_context

def collect_engines(script_module):
    """
    Collect all engines in a script_module of disc
    """
    return utils.collect_engines(script_module, _DISC_GROUP_NAME)


def num_engines(script_module):
    """
    Return the number of engines of MLIR
    """
    return utils.num_engines(script_module, _DISC_GROUP_NAME)

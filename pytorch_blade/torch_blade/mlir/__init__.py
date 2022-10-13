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
    from torch_blade.config import OptPipelines
    from torch_blade.mlir.disc_engine_conversion import _compile_torchscript, _optimize_mlir

    from .._torch_blade._mlir import *

    _DISC_NAME = backend_name()
    _is_available = True
    _DISC_GROUP_NAME = _DISC_NAME.lower() + "_grp"
    OptPipelines.register_pipeline(_DISC_NAME, _optimize_mlir)
except ImportError as e:
    from torch_blade.logging import logger
    logger.warning(e)
    # MLIR support is disable
    _is_available = False
    _DISC_GROUP_NAME = None
    _DISC_NAME = "DISC"

import contextlib

from torch_blade import utils

_DISC_TESTING_CONTEXT = False

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


def num_compiled_nodes(script_module):
    """
    Return the number of nodes compiled by MLIR
    """
    return utils.num_compiled_nodes(script_module, _DISC_GROUP_NAME)

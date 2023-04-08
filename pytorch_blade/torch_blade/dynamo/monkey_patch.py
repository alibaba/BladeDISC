# Copyright 2023 The BladeDISC Authors. All rights reserved.
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
from torch import Tensor
from torch._decomp import register_decomposition
from typing import Callable, cast, Iterable, List, Optional, Tuple, Union

from torch._dynamo.symbolic_convert import InstructionTranslatorBase
from torch._dynamo.variables.base import VariableTracker, MutableLocal
from torch._dynamo.variables.dicts import ConstDictVariable

# NOTE: this is a temporary monkey patch to support tracing unpacked kwargs,
# let's create a pull request to pytorch in the future,
# the related issue: https://github.com/pytorch/pytorch/issues/91274
def BUILD_MAP_UNPACK_WITH_CALL(self, inst):
    kwargs = self.popn(inst.argval)
    options = VariableTracker.propagate(kwargs)
    results = dict()
    for arg in kwargs:
        for k, v in arg.items.items():
            results[k] = v

    if len(results) == 0: return
    self.push(
        ConstDictVariable(results, dict, mutable_local=MutableLocal(), **options)
    )

if not hasattr(InstructionTranslatorBase, "BUILD_MAP_UNPACK_WITH_CALL"):
    InstructionTranslatorBase.BUILD_MAP_UNPACK_WITH_CALL = BUILD_MAP_UNPACK_WITH_CALL

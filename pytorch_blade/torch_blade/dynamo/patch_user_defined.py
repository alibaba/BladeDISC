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

import functools
import torch._dynamo.variables as variables
import torch._dynamo.variables.dicts as dicts
from torch._dynamo.eval_frame import skip_code

_DataClassVariable = dicts.DataClassVariable

class PatchedDataClassVariable(_DataClassVariable):
    @staticmethod
    @functools.lru_cache(None)
    def _patch_once():
        _DataClassVariable._patch_once()
        from diffusers.models.attention import Transformer2DModelOutput

        for obj in Transformer2DModelOutput.__dict__.values():
            if callable(obj):
                skip_code(obj.__code__)

    @staticmethod
    def is_matching_cls(cls):
        if _DataClassVariable.is_matching_cls(cls):
            return True
        try:
            from diffusers.models.attention import Transformer2DModelOutput

            return issubclass(cls, Transformer2DModelOutput)
        except ImportError:
            return False

variables.DataClassVariable = PatchedDataClassVariable


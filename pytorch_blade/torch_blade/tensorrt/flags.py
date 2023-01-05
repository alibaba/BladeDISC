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

import contextlib

try:
    from .._torch_blade._tensorrt import *
except ImportError:
    pass


@contextlib.contextmanager
def builder_flags_context(flags):
    old_flags = set_builder_flags(flags)
    try:
        yield
    finally:
        set_builder_flags(old_flags)

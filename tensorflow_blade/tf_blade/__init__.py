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

# flake8: noqa

import os as _os

from tf_blade.util.tf_import_helper import tf

# Load libtf_blade.so before import _tf_blade.so to work around differnent behavior
# between TF 1.x and 2.x. In TF 1.x, there are op registery in both Python and C++
# side. If a shared library is loaded via dynamic linker, the ops won't be registered
# into the registery in Python side. While in TF 2.x, Python side share's registery oon
# C++ side.
tf.load_op_library(_os.path.join(_os.path.dirname(__file__), 'libtf_blade.so'))

from ._tf_blade import *

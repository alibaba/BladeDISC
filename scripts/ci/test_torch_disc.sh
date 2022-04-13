#!/bin/bash
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
set -e
pushd pytorch_blade
# 1. using a virtualenv to avoid permission issue
python -m virtualenv --system-site-packages myenv && source myenv/bin/activate

# 2. build BladeDISC with ltc
python setup.py develop

# 3. test a e2e demo
python -m unittest discover torch_disc/python/tests

deactivate
popd

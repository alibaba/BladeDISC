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

# 1. using a virtualenv to avoid permission issue
python -m virtualenv --system-site-packages myenv && source myenv/bin/activate

# 2. common setup for the project
python scripts/python/common_setup.py
# 3. run unit tests and build _torch_disc.so
cd pytorch_blade
python setup.py cpp_test
python setup.py develop
# 4. test a e2e demo
pytest tests/ltc -v -m "ltc"

deactivate

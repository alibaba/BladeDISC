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

python scripts/python/tao_build.py /opt/venv_disc -s configure --bridge-gcc default --compiler-gcc default

python -m virtualenv --system-site-packages myenv && source myenv/bin/activate
# build _torch_disc.so
(cd torch_disc && python setup.py develop)
# check pybind library
(cd torch_disc/bazel-bin/torch_disc && python -c "import _torch_disc")
deactive

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

# 1. configure tensorflow
python scripts/python/tao_build.py /opt/venv_disc -s configure --bridge-gcc default --compiler-gcc default
python scripts/python/tao_build.py /opt/venv_disc -s build_mlir_ral
ln -sf /workspace/tf_community/bazel-bin/tensorflow/compiler/mlir/disc/disc_compiler_main torch_disc/disc_compiler_main
# 2. using a virtualenv to avoid permission issue
python -m virtualenv --system-site-packages myenv && source myenv/bin/activate
# 3. run unit tests and build _torch_disc.so
cd torch_disc
python setup.py test
python setup.py develop
# 4. test a e2e demo
ln -sf bazel-bin/torch_disc/_torch_disc.so ./_torch_disc.so
ln -sf bazel-out/k8-opt/bin/external/org_tensorflow/tensorflow/compiler/mlir/xla/ral/libral_base_context.so ./libral_base_context.so
python -m unittest discover torch_disc/python/tests
python disc_demo.py

deactivate

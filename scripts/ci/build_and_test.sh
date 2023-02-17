#!/bin/bash
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


set -ex

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

ENTRY=scripts/python/tao_build.py
VENV_PATH=/opt/venv_disc
BLADE_DISC_DIR=tao/python/blade_disc_tf

source ${SCRIPT_DIR}/parse_args.sh "$@"

if [[ -f ~/.cache/proxy_config ]]; then
    source ~/.cache/proxy_config
fi

# cleanup build cache
(rm -rf build \
  && rm -rf tao/build \
  && cd tao && bazel clean --expunge && cd ..\
  && cd tao_compiler && bazel clean --expunge)

python ${ENTRY} ${VENV_PATH} -s configure --bridge-gcc default --compiler-gcc default ${CPU_ONLY} ${ROCM} ${DCU} ${ROCM_PATH} ${TARGET_CPU_ARCH}
python ${ENTRY} ${VENV_PATH} -s build_tao_bridge ${CPU_ONLY} ${ROCM} ${DCU} ${ROCM_PATH} ${TARGET_CPU_ARCH}
python ${ENTRY} ${VENV_PATH} -s build_tao_compiler ${CPU_ONLY} ${ROCM} ${DCU} ${ROCM_PATH} ${TARGET_CPU_ARCH}
if [[ -z "$ROCM" ]] && [[ -z "$DCU" ]]; then
  python ${ENTRY} ${VENV_PATH} -s test_tao_bridge_cpp ${CPU_ONLY} ${ROCM} ${DCU} ${ROCM_PATH}
  python ${ENTRY} ${VENV_PATH} -s test_tao_bridge_py ${CPU_ONLY} ${ROCM} ${DCU} ${ROCM_PATH}
  python ${ENTRY} ${VENV_PATH} -s test_tao_compiler ${CPU_ONLY} ${ROCM} ${DCU} ${ROCM_PATH}
fi

# copy libtao_ops.so and tao_compiler_main to blade-disc-tf
cp tao/bazel-bin/libtao_ops.so ${BLADE_DISC_DIR}
cp tao_compiler/bazel-bin/decoupling/tao_compiler_main ${BLADE_DISC_DIR}

if [[ -n "$ROCM" ]] || [[ -n "$DCU" ]]; then
  # TODO: skip the following stages if rocm build
  exit 0
fi

(cd tao && \
  ${VENV_PATH}/bin/pytest --pyargs python
  ${VENV_PATH}/bin/python setup.py bdist_wheel)

# copy Python wheel package to build folder
mkdir -p build && \
cp tao/dist/blade_disc*.whl ./build
cp tao_compiler/bazel-bin/mlir/disc/tools/disc-replay/disc-replay-main ./build/

# test example models
if [[ -z "$ROCM" ]] && [[ -z "$DCU" ]]; then
  source ${SCRIPT_DIR}/test_cpu_examples.sh
fi

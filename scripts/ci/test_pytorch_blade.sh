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
# bazel cache
export CXXFLAGS=${CXXFLAGS:-"-Wno-deprecated-dewlarations"}
export CFLAGS=${CFLAGS:-"-Wno-deprecated-declarations"}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CUDACXX=${CUDACXX:-"${CUDA_HOME}/bin/nvcc"}
export PATH=${CUDA_HOME}/bin/:$PATH
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=${CUDA_HOME}/lib64:$LIBRARY_PATH
export TF_REMOTE_CACHE=${TF_REMOTE_CACHE}

# cleanup build cache
(cd tf_community && bazel clean --expunge)

# note(yancey.yx): using virtualenv to avoid permission issue on workflow actions CI,
python -m virtualenv venv && source venv/bin/activate

export TORCH_BLADE_CI_BUILD_TORCH_VERSION=${TORCH_BLADE_CI_BUILD_TORCH_VERSION:-1.7.1+cu110}
(cd pytorch_blade \
  && python -m pip install -q -r requirements-dev-${TORCH_BLADE_CI_BUILD_TORCH_VERSION}.txt \
       -f https://pai-blade.oss-cn-zhangjiakou.aliyuncs.com/pytorch/wheels/repo.html \
  && TORCH_LIB=$(python -c 'import torch; import os; print(os.path.dirname(os.path.abspath(torch.__file__)) + "/lib/")') \
  && export LD_LIBRARY_PATH=$TORCH_LIB:$LD_LIBRARY_PATH \
  && bash ./ci_build/build_pytorch_blade.sh)

mkdir -p build && \
mv pytorch_blade/dist/torch_blade*.whl ./build

deactivate

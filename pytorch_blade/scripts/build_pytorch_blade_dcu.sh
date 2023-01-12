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

# !/bin/bash
set -o pipefail
set -e
export ROCM_HOME=${ROCM_HOME:-/opt/dtk}
export TF_ROCM_HOME=${ROCM_HOME}
export ROCMCXX=${ROCMCXX:-"${ROCM_HOME}/bin/hipcc"}
export PATH=${ROCM_HOME}/bin/:$PATH
# Build TorchBlade with DEBUG
# export DEBUG=1
export TORCH_BLADE_USE_ROCM=${TORCH_BLADE_USE_ROCM:-ON}
export TORCH_BLADE_BUILD_MLIR_SUPPORT=${TORCH_BLADE_BUILD_MLIR_SUPPORT:-ON}
export TORCH_BLADE_BUILD_WITH_CUDA_SUPPORT=${TORCH_BLADE_BUILD_WITH_CUDA_SUPPORT:-OFF}
export TORCH_BLADE_BUILD_WITH_DCU_ROCM_SUPPORT=${TORCH_BLADE_BUILD_WITH_DCU_ROCM_SUPPORT:-ON}
export TORCH_BLADE_DISABLE_PATCHELF_CUDA_SONAMES=${TORCH_BLADE_DISABLE_PATCHELF_CUDA_SONAMES:-OFF} 
 
function ci_build() {
    echo "DO TORCH_BLADE CI_BUILD"
    if [ "$TORCH_BLADE_BUILD_WITH_CUDA_SUPPORT" = "ON"  ]; then
      export TORCH_BLADE_BUILD_TENSORRT=ON
      export TORCH_BLADE_BUILD_TENSORRT_STATIC=${TORCH_BLADE_BUILD_TENSORRT_STATIC:-OFF}
      python3 ../scripts/python/common_setup.py
    elif [ "$TORCH_BLADE_BUILD_WITH_DCU_ROCM_SUPPORT" = "ON"  ]; then
      export TORCH_BLADE_BUILD_TENSORRT=OFF
      export TORCH_BLADE_BUILD_TENSORRT_STATIC=${TORCH_BLADE_BUILD_TENSORRT_STATIC:-OFF}
      python3 ../scripts/python/common_setup.py --rocm_path=/opt/dtk 
    else
      python3 ../scripts/python/common_setup.py --cpu_only
    fi
    TORCH_LIB=$(python -c 'import torch; import os; print(os.path.dirname(os.path.abspath(torch.__file__)) + "/lib/")') \
    export LD_LIBRARY_PATH=$TORCH_LIB:$LD_LIBRARY_PATH \
    export TORCH_BLADE_SKIP_DISC_CMD_BUILD=OFF
    rm -rf build && python3 setup.py develop;
    export TORCH_BLADE_DEBUG_LOG=ON
    python3 setup.py bdist_wheel;
}
 
# Build
ci_build

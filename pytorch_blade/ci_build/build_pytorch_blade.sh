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

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda/}
export CUDACXX=${CUDACXX:-"${CUDA_HOME}/bin/nvcc"}
export PATH=${CUDA_HOME}/bin/:$PATH
export TENSORRT_INSTALL_PATH=${TENSORRT_INSTALL_PATH:-/usr/local/TensorRT/}
export LD_LIBRARY_PATH=${TENSORRT_INSTALL_PATH}/lib/:${TENSORRT_INSTALL_PATH}/lib64/:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=${TENSORRT_INSTALL_PATH}/lib/:${TENSORRT_INSTALL_PATH}/lib64/:${CUDA_HOME}/lib64:$LIBRARY_PATH

# Build TorchBlade with DEBUG
# export DEBUG=1
export TORCH_BLADE_BUILD_MLIR_SUPPORT=${TORCH_BLADE_BUILD_MLIR_SUPPORT:-ON}
export TORCH_BLADE_BUILD_WITH_CUDA_SUPPORT=${TORCH_BLADE_BUILD_WITH_CUDA_SUPPORT:-ON}

function pip_install_deps() {
    # set TORCH_BLADE_CI_BUILD_TORCH_VERSION default to 1.7.1+cu110
    TORCH_BLADE_CI_BUILD_TORCH_VERSION=${TORCH_BLADE_CI_BUILD_TORCH_VERSION:-1.7.1+cu110}
    requirements=requirements-dev-${TORCH_BLADE_CI_BUILD_TORCH_VERSION}.txt
    python3 -m pip install --upgrade pip
    python3 -m pip install cmake ninja virtualenv
    python3 -m pip install -r ${requirements} -f https://download.pytorch.org/whl/torch_stable.html
}

function ci_build() {
    echo "DO TORCH_BLADE CI_BUILD"
    pip_install_deps

    if [ "$TORCH_BLADE_USE_CMAKE_BUILD" = "ON"  ]; then
      extra_args="--cmake"
    else
      extra_args=""
      if [ "$TORCH_BLADE_USE_CMAKE_BUILD" = "ON"  ]; then
        python3 ../scripts/python/common_setup.py
      else
        python3 ../scripts/python/common_setup.py --cpu_only
      fi
    fi

    export TORCH_BLADE_BUILD_TENSORRT=ON
    rm -rf build && python3 setup.py develop ${extra_args};
    # The following are UNIT TESTS
    export TORCH_BLADE_DEBUG_LOG=ON
    python3 setup.py cpp_test ${extra_args} 2>&1 | tee -a cpp_test.out;
    python3 -m unittest discover tests/ -v 2>&1 | tee -a py_test.out;
    python3 setup.py bdist_wheel ${extra_args};
}

# Build
ci_build

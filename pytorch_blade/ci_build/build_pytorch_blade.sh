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
# bazel cache
export CXXFLAGS=${CXXFLAGS:-"-Wno-deprecated-dewlarations"}
export CFLAGS=${CFLAGS:-"-Wno-deprecated-declarations"}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda/}
export CUDACXX=${CUDACXX:-"${CUDA_HOME}/bin/nvcc"}
export PATH=${CUDA_HOME}/bin/:$PATH
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=${CUDA_HOME}/lib64:$LIBRARY_PATH
export GCC_HOST_COMPILER_PATH=$(which gcc) # needed by bazel crosstool

# Build TorchBlade with DEBUG
# export DEBUG=1

# To save time, set USE_BLADE_DISC_PRE_BUILD=ON if you has already built blade_disc
# export USE_BLADE_DISC_PRE_BUILD=ON

export TORCH_BLADE_BUILD_MLIR_SUPPORT=${TORCH_BLADE_BUILD_MLIR_SUPPORT:-ON}
export TORCH_BLADE_BUILD_WITH_CUDA_SUPPORT=${TORCH_BLADE_BUILD_WITH_CUDA_SUPPORT:-ON}

function pip_install_deps() {
    echo "DO TORCH_BLADE CI_BUILD"
    # set TORCH_BLADE_CI_BUILD_TORCH_VERSION default to 1.7.1+cu110
    TORCH_BLADE_CI_BUILD_TORCH_VERSION=${TORCH_BLADE_CI_BUILD_TORCH_VERSION:-1.7.1+cu110}
    requirements=requirements-dev-${TORCH_BLADE_CI_BUILD_TORCH_VERSION}.txt
    python3 -m pip install --upgrade pip
    python3 -m pip install cmake ninja virtualenv
    python3 -m pip install -r ${requirements} -f https://download.pytorch.org/whl/torch_stable.html
}

function bazel_build() {
    python3 ../scripts/python/common_setup.py
    rm -rf build && python3 setup.py develop;
}

function cmake_build() {
    rm -rf build && python3 setup.py develop --cmake;
}

function ci_build() {
    pip_install_deps

    if [ "$TORCH_BLADE_USE_CMAKE_BUILD" = "ON"  ]; then
      cmake_build
    else
      bazel_build
    fi

    # The following are UNIT TESTS
    export TORCH_BLADE_DEBUG_LOG=ON
    python3 setup.py cpp_test 2>&1 | tee -a cpp_test.out;
    python3 -m unittest discover tests/ -v 2>&1 | tee -a py_test.out;
    python3 setup.py bdist_wheel;
}

# Build
ci_build

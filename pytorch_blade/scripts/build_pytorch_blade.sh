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
export TF_CUDA_HOME=${CUDA_HOME} # for cuda_supplement_configure.bzl
export CUDACXX=${CUDACXX:-"${CUDA_HOME}/bin/nvcc"}
export PATH=${CUDA_HOME}/bin/:$PATH
export TENSORRT_INSTALL_PATH=${TENSORRT_INSTALL_PATH:-/usr/local/TensorRT/}
export LD_LIBRARY_PATH=${TENSORRT_INSTALL_PATH}/lib/:${TENSORRT_INSTALL_PATH}/lib64/:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=${TENSORRT_INSTALL_PATH}/lib/:${TENSORRT_INSTALL_PATH}/lib64/:${CUDA_HOME}/lib64:$LIBRARY_PATH

# Build TorchBlade with DEBUG
# export DEBUG=1
export TORCH_BLADE_BUILD_MLIR_SUPPORT=${TORCH_BLADE_BUILD_MLIR_SUPPORT:-ON}
export TORCH_BLADE_BUILD_WITH_CUDA_SUPPORT=${TORCH_BLADE_BUILD_WITH_CUDA_SUPPORT:-ON}
export TORCH_BLADE_RUN_EXAMPLES=${TORCH_BLADE_RUN_EXAMPLES:-OFF}

function pip_install_deps() {
    # set TORCH_BLADE_CI_BUILD_TORCH_VERSION default to 1.7.1+cu110
    TORCH_BLADE_CI_BUILD_TORCH_VERSION=${TORCH_BLADE_CI_BUILD_TORCH_VERSION:-1.7.1+cu110}
    requirements=requirements-dev-${TORCH_BLADE_CI_BUILD_TORCH_VERSION}.txt
    python3 -m pip install --upgrade pip
    python3 -m pip install virtualenv
    python3 -m pip install -r scripts/pip/${requirements} -f https://download.pytorch.org/whl/torch_stable.html
}

function ci_build() {
    echo "DO TORCH_BLADE CI_BUILD"
    pip_install_deps

    COMMON_SETUP_ARGS=""
    if [ "$TORCH_BLADE_USE_PLATFORM_ALIBABA" = "ON"  ]; then
      COMMON_SETUP_ARGS="--platform_alibaba"
    fi

    if [ "$TORCH_BLADE_BUILD_ON_YITIAN" = "ON" ]; then
      COMMON_SETUP_ARGS+=" --target_cpu_arch='armv8.6-a-sve2'"
    fi

    if [ "$TORCH_BLADE_BUILD_WITH_CUDA_SUPPORT" = "ON"  ]; then
      export TORCH_BLADE_BUILD_TENSORRT=${TORCH_BLADE_BUILD_TENSORRT:-ON}
      export TORCH_BLADE_BUILD_TENSORRT_STATIC=${TORCH_BLADE_BUILD_TENSORRT_STATIC:-OFF}
      python3 ../scripts/python/common_setup.py $COMMON_SETUP_ARGS
    else
      python3 ../scripts/python/common_setup.py --cpu_only $COMMON_SETUP_ARGS
    fi
    TORCH_LIB=$(python -c 'import torch; import os; print(os.path.dirname(os.path.abspath(torch.__file__)) + "/lib/")') \
    export LD_LIBRARY_PATH=$TORCH_LIB:$LD_LIBRARY_PATH

    # DEBUG=1 will trigger debug mode compilation
    DEBUG=1 python3 setup.py cpp_test 2>&1 | tee cpp_test.out;

    export TORCH_BLADE_SKIP_DISC_CMD_BUILD=OFF
    rm -rf build && python3 setup.py develop;
    # The following are UNIT TESTS
    export TORCH_BLADE_DEBUG_LOG=ON
    # disable tf32 on A100
    export NVIDIA_TF32_OVERRIDE=0
    # neural_engine is under development, the uts are only enabled
    # when it is available.
    if [ "$TORCH_BLADE_ENABLE_NEURAL_ENGINE" != "ON"  ]; then
      PYTEST_EXCLUDED_TESTS="tests/neural_engine"
    else
      PYTEST_EXCLUDED_TESTS=""
    fi
    TORCH_DISC_USE_TORCH_MLIR=true pytest tests -v --ignore=$PYTEST_EXCLUDED_TESTS 2>&1 | tee -a py_test.out
    python3 setup.py bdist_wheel;
}

function test_training_examples() {
  if [ "$TORCH_BLADE_CI_BUILD_TORCH_VERSION" == "pre+cu117" ]; then
    (
      cd ../examples/PyTorch/Train/Dynamo/Bert && \
      python3 -m pip install -r requirements.txt && \
      bash ./test.sh
    )
  fi
}

function test_cpu_infer_examples() {
  pushd ../examples/PyTorch/Inference/CPU
  bash albert/test.sh
  popd
}

function test_cuda_infer_examples() {
  pushd ../examples/PyTorch/Inference/CUDA
  bash BERT/test.sh
  bash ResNet/test.sh
  bash S2T/test.sh
  bash T5/test.sh
  popd
}

# Build
ci_build
if [ "$TORCH_BLADE_RUN_EXAMPLES" == "ON" ]; then
  if [ "$TORCH_BLADE_BUILD_WITH_CUDA_SUPPORT" == "ON" ]; then
    test_cuda_infer_examples
    test_training_examples
  else
    test_cpu_infer_examples
  fi
fi

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
set -e
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda/}
export TF_CUDA_HOME=${CUDA_HOME} # for cuda_supplement_configure.bzl
export CUDACXX=${CUDACXX:-"${CUDA_HOME}/bin/nvcc"}
export PATH=${CUDA_HOME}/bin/:$PATH
export TENSORRT_INSTALL_PATH=${TENSORRT_INSTALL_PATH:-/usr/local/TensorRT/}

# Build TorchBlade with DEBUG
# export DEBUG=1
export TORCH_BLADE_BUILD_MLIR_SUPPORT=${TORCH_BLADE_BUILD_MLIR_SUPPORT:-ON}
export TORCH_BLADE_BUILD_WITH_CUDA_SUPPORT=${TORCH_BLADE_BUILD_WITH_CUDA_SUPPORT:-ON}

function build_torch_blade() {
    python3 ../scripts/python/common_setup.py

    export TORCH_BLADE_BUILD_TENSORRT=ON
    rm -rf build && python3 setup.py develop;
    # The following are UNIT TESTS
    export TORCH_BLADE_DEBUG_LOG=ON
    python3 setup.py cpp_test 2>&1 | tee -a cpp_test.out;
    python3 setup.py bdist_wheel;
}

script_dir=$(cd $(dirname "$0"); pwd)
# step in TorchBlade root dir
pushd $script_dir/..
echo DIR: $(pwd)

# Build
python3 -m virtualenv venv --system-site-packages && source venv/bin/activate
build_torch_blade
python3 -m pip install -r benchmark/requirements.txt
bash benchmark/detectron2/test_d2_benchmark.sh 2>&1 | tee test_d2.log
bash benchmark/torch-tensorrt/test_trt_benchmark.sh 2>&1 | tee test_trt.log

grep "|" test_d2.log
grep "|" test_trt.log

popd # $script_dir/..

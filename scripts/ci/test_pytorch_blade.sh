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
export CXXFLAGS=${CXXFLAGS:-"-Wno-deprecated-declarations"}
export CFLAGS=${CFLAGS:-"-Wno-deprecated-declarations"}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export TF_CUDA_HOME=${CUDA_HOME}
export CUDACXX=${CUDACXX:-"${CUDA_HOME}/bin/nvcc"}
export PATH=${CUDA_HOME}/bin/:$PATH
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=${CUDA_HOME}/lib64:$LIBRARY_PATH
export TF_REMOTE_CACHE=${TF_REMOTE_CACHE}
export TORCH_BLADE_BUILD_TENSORRT_STATIC=${TORCH_BLADE_BUILD_TENSORRT_STATIC:-OFF}

function delete_disk_caches() {
  # https://bazel.build/remote/caching#delete-remote-cache
  ndays=10
  if [ -d "cas" ] && [ -d "ac" ]; then
    echo "Before old disk caches deleting"
    du cas/ ac/ --max-depth=0 -h
    find cas -type f -mtime +${ndays} -delete
    find ac -type f -mtime +${ndays} -delete
    echo "After old disk caches deleting"
    du cas/ ac/ --max-depth=0 -h
  fi
}

(cd ~/.cache/ && delete_disk_caches)

if [[ -f ~/.cache/proxy_config ]]; then
  source ~/.cache/proxy_config
fi

# cleanup build cache
(cd tao_compiler && bazel clean --expunge)

# note(yancey.yx): using virtualenv to avoid permission issue on workflow actions CI,
if [ $TORCH_BLADE_CI_BUILD_TORCH_VERSION = "ngc" ]; then
  python -m virtualenv venv --system-site-packages && source venv/bin/activate
else
  python -m virtualenv venv && source venv/bin/activate
fi

python -m pip install -U pip
python -m pip install onnx==1.11.0 # workaround for latest onnx installing failure

arch=`uname -p`
if [[ $arch == "aarch64" ]]; then
  # higher prootbuf version has compatible problem.
  # TODO(disc): upgrade protobuf once we fix the problem.
  python -m pip install protobuf==3.20.1
fi

export TORCH_BLADE_CI_BUILD_TORCH_VERSION=${TORCH_BLADE_CI_BUILD_TORCH_VERSION:-1.7.1+cu110}
(cd pytorch_blade && bazel clean --expunge \
  && bash ./scripts/build_pytorch_blade.sh)

mkdir -p build && \
mv pytorch_blade/dist/torch_blade*.whl ./build

case $TORCH_BLADE_CI_BUILD_TORCH_VERSION in
  *1.11.* | *1.12.* | *1.13.*)
    (cd tools/torch_quant && bash ./test.sh)
    mv tools/torch_quant/dist/torch_quant*.whl ./build
    ;;
esac

deactivate

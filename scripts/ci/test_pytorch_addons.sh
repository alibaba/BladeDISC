#!/bin/bash
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

(cd pytorch_blade \
  && python -m pip install -q -r requirements-dev-1.7.1+cu110.txt -f https://pai-blade.oss-cn-zhangjiakou.aliyuncs.com/pytorch/wheels/repo.html  \
  && TORCH_LIB=$(python -c 'import torch; import os; print(os.path.dirname(os.path.abspath(torch.__file__)) + "/lib/")') \
  && export LD_LIBRARY_PATH=$TORCH_LIB:$LD_LIBRARY_PATH \
  && bash ./ci_build/build_pytorch_blade.sh)

# build TorchBlade Python pacakge
(cd pytorch_blade \
  && python setup.py bdist_wheel)

mkdir -p build && \
mv pytorch_blade/dist/torch_blade*.whl ./build

deactivate

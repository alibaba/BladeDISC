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

script_dir=$(cd $(dirname "$0"); pwd)
# step in script dir
pushd $script_dir
echo DIR: $(pwd)

export PYTORCH_JIT_LOG_LEVEL=">>>mhlo_conversion.cpp"
export TENSORRT_INSTALL_PATH=${TENSORRT_INSTALL_PATH:-/usr/local/TensorRT/}
export LD_LIBRARY_PATH=${TENSORRT_INSTALL_PATH}/lib/:${TENSORRT_INSTALL_PATH}/lib64/:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
pip3 install -r requirements.txt
python3 main.py --disc-only
popd

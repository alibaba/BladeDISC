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

set -ex
if [ ! -z "$CPU_ONLY" ]; then
  if [[ -f ~/.cache/proxy_config ]]; then
    source ~/.cache/proxy_config
  fi
  # install disc python wheel
  python -m virtualenv venv && source venv/bin/activate
  arch=`uname -p`
  if [[ $arch == "aarch64" ]]; then
    # higher version is not backward compatible.
    pip install protobuf==3.20.1
    # TODO(disc): a workaround for issue #224
    pip install https://pai-blade.oss-accelerate.aliyuncs.com/build_deps/tensorflow/2.8.0_py3.8.5_aarch64/tensorflow-2.8.0-cp38-cp38-linux_aarch64.whl
  fi
  HTTPS_PROXY=${HTTPS_PROXY} python -m pip install ./build/blade_disc*.whl

  pushd examples/TensorFlow/Inference/X86/BERT
  bash download_model.sh
  TAO_ENABLE_FALLBACK=false python main.py
  # clean up download files
  rm -rf model
  popd
  rm -rf venv
fi

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
python3 -m pip install virtualenv numpy
DISC_VENV=/opt/venv_disc

function install_venv() {
  python3 -m virtualenv ${DISC_VENV}
  source ${DISC_VENV}/bin/activate
  # TODO(disc): tensorflow-aarch64 wheel is not compatible with `libtao_ops.so`.
  # Thus we use the wheel package provided by the ARM:
  #   https://github.com/ARM-software/Tool-Solutions/tree/master/docker/tensorflow-aarch64
  # See #244 for more details.
  # if [[ ! -z "${DISC_HOST_TF_VERSION}" ]]; then
  #   echo "install TensorFlow: "${DISC_HOST_TF_VERSION} "..."
  #   pip install -q ${DISC_HOST_TF_VERSION}
  # fi

  # higher version is not backward compatible.
  pip install protobuf==3.20.1
  pip install https://pai-blade.oss-accelerate.aliyuncs.com/build_deps/tensorflow/2.8.0_py3.8.5_aarch64/tensorflow-2.8.0-cp38-cp38-linux_aarch64.whl

  pip install pytest pytest-forked
  deactivate
}

install_venv

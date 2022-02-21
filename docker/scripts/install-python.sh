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

function install_tf115() {
  # install virtualenv: Python3+TensorFlow-GPU 1.15
  pip install -q tensorflow-gpu==1.15
}

function install_tf24() {
  # install virtualenv: Python3+TensorFlow-GPU 2.4.0
  pip install -q tensorflow-gpu==2.4
}

function install_tf115_cpu() {
  # install virtualenv: Python3+TensorFlow 1.15
  pip install -q tensorflow==1.15
}

function install_venv() {
  python3 -m virtualenv ${DISC_VENV}
  source ${DISC_VENV}/bin/activate
  if [[ ! -z "${BLADE_DISC_BUILT_CPU}" ]]; then
    install_tf115_cpu
  elif [[ "$CUDA_VERSION" == 10.0.* ]]; then
    # CUDA10 Docker image
    install_tf115
  elif [[ "$CUDA_VERSION" == 11.0.* ]]; then
    install_tf24
    # CUDA11 Docker image
  else
    echo "unsupported CUDA version: " $CUDA_VERSION
    exit -1
  fi
  pip install pytest pytest-forked
  deactivate
}

install_venv

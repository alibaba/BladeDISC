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


set -e
PYTHON_VERSION=${PYTHON_VERSION:-PYTHON3.6}
DISC_VENV=/opt/venv_disc

function install_python() {
  if [[ "$PYTHON_VERSION" == "PYTHON3.6" ]]; then
    apt-get install -y python3.6 python3.6-dev python3-pip
  elif [[ "$PYTHON_VERSION" == "PYTHON3.8" ]]; then
    apt-get install -y python3.8 python3.8-dev python3-pip
    rm -rf /usr/bin/python3
    ln -s /usr/bin/python3.8 /usr/bin/python3
  else
    echo "PYTHON_VERSION should be in [PYTHON3.6, PYTHON3.8]"
    exit 1
  fi
  ln -s /usr/bin/python3 /usr/bin/python
  python3 -m pip install --upgrade pip
  python3 -m pip install virtualenv numpy
}

function install_venv() {
  python3 -m virtualenv ${DISC_VENV}
  source ${DISC_VENV}/bin/activate
  if [[ ! -z "${DISC_HOST_TF_VERSION}" ]]; then
    echo "install TensorFlow: "${DISC_HOST_TF_VERSION} "..."
    pip install -q ${DISC_HOST_TF_VERSION}
  fi

  pip install pytest pytest-forked
  deactivate
}

install_python
install_venv

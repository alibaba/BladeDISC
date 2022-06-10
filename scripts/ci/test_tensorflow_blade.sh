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

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <device> <tf_version>"
    exit -1
fi

device=$1
tf_ver=$2

# RuntimeError: Click will abort further execution because Python was configured to use ASCII as encoding for the environment. Consult https://click.palletsprojects.com/unicode-support/ for mitigation steps.
# This system supports the C.UTF-8 locale which is recommended. You might be able to resolve your issue by exporting the following environment variables:
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

VENV_PATH=venv
# note(yancey.yx): using virtualenv to avoid permission issue on workflow actions CI,
python -m virtualenv ${VENV_PATH} && source ${VENV_PATH}/bin/activate
python -m pip install -U pip
python -m pip install onnx==1.11.0 # workaround for latest onnx installing failure


pushd tensorflow_blade

require_txt="requirement-tf${tf_ver}-${device}.txt"
[[ ! -f ${require_txt} ]] && echo "requirement-tf${tf_ver}-${device}.txt not found" && exit -1
python -m pip install -q -r ${require_txt}

device_type="gpu"
[[ ${device} == "cpu" ]] && device_type="cpu"

./build.py -s configure --device ${device_type}
./build.py -s check
./build.py -s build
./build.py -s develop
./build.py -s test
./build.py -s package
popd

mkdir -p build && \
mv tensorflow_blade/dist/tensorflow_blade*.whl ./build

deactivate

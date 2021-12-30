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
function help {
    echo "Usage: $0 --py_ver PYTHON_VERSION [--ral_cxx11_abi] [--cpu_only] [--dcu]"
}

function require {
    value=$1
    flag=$2
    if [[ "${value}" == "" ]]; then
        echo "Required flag is missing: ${flag}"
        help
        exit -1;
    fi
}

ARGS=`getopt -o h --long py_ver:,ral_cxx11_abi,dcu,cpu_only -n "$0" -- "$@"`
if [ $? != 0 ]; then
    echo "Error when preparing arguments..."
    exit 1
fi

eval set -- "${ARGS}"

PYTHON_VERSION=""
BLADE_DISC_CXX11_ABI_CONFIG=""
BLADE_DISC_CPU_ONLY_CONFIG=""
BLADE_DISC_DCU_CONFIG=""

while true
do
    case "$1" in
        --py_ver)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --ral_cxx11_abi)
            BLADE_DISC_CXX11_ABI_CONFIG="--ral_cxx11_abi"
            shift
            ;;
        --dcu)
            BLADE_DISC_DCU_CONFIG="--dcu"
            shift
            ;;
        --cpu_only)
            BLADE_DISC_CPU_ONLY_CONFIG="--cpu_only"
            shift
            ;;
        -h)
            help
            exit
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Internal error!"
            exit 1
            ;;
    esac
done

require "${PYTHON_VERSION}" "--py_ver"

echo "PYTHON_VERSION=${PYTHON_VERSION}"
echo "BLADE_DISC_CXX11_ABI_CONFIG=${BLADE_DISC_CXX11_ABI_CONFIG}"
echo "BLADE_DISC_CPU_ONLY_CONFIG=${BLADE_DISC_CPU_ONLY_CONFIG}"
echo "BLADE_DISC_DCU_CONFIG=${BLADE_DISC_DCU_CONFIG}"

CI_SCRIPTS_DIR=./scripts/python/

# Currently TVM is not needed and would break build_mlir_ral
export TF_NEED_TVM=0
BLADE_DISC_VENV_DIR=$(dirname $(dirname $(which ${PYTHON_VERSION})))
${CI_SCRIPTS_DIR}/tao_build.py ${BLADE_DISC_VENV_DIR} --compiler-gcc default --bridge-gcc default --stage configure_pytorch ${BLADE_DISC_CXX11_ABI_CONFIG} ${BLADE_DISC_CPU_ONLY_CONFIG} ${BLADE_DISC_DCU_CONFIG}
${CI_SCRIPTS_DIR}/tao_build.py ${BLADE_DISC_VENV_DIR} --stage build_mlir_ral ${BLADE_DISC_CXX11_ABI_CONFIG} ${BLADE_DISC_CPU_ONLY_CONFIG} ${BLADE_DISC_DCU_CONFIG}


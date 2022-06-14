#!/bin/bash
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



function log {
    now=`date '+%Y-%m-%d %H:%M:%S.%3N'`
    if [ -t 1 ]; then
        echo -e "${now} \033[32m\e[1m [INFO]\e[0m $@"
    else
        echo -e "${now} [INFO] $@"
    fi
}

function usage {
    echo "$0 --device cpu|gpu [--skip_trt] [--develop]"
}


set -e

ARGS=`getopt -o h --long py_bin:,device:,skip_trt,develop -n "$0" -- "$@"`
if [ $? != 0 ]; then
    usage
    exit 1
fi
eval set -- "${ARGS}"

export SKIP_TRT=false
export PKG_DEVICE=gpu
py_bin=""
is_develop='0'

while true; do
    case "$1" in
        --py_bin)
            py_bin=$2
            shift 2
            ;;
        --device)
            export PKG_DEVICE=$2
            shift 2
            ;;
        --skip_trt)
            export SKIP_TRT=true
            shift 1
            ;;
        --develop)
            is_develop='1'
            shift 1
            ;;
        -h)
            usage
            exit 0
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

[[ -z ${py_bin} ]] && echo "--py_bin is required!" && exit 1

log "=================== Build Arguments ==================="
log "Current dir      : $(pwd)"
log "PKG_DEVICE       : ${PKG_DEVICE}"
log "SKIP_TRT         : ${SKIP_TRT}"
log "Python binary    : ${py_bin}"
log "Is Develop Mode? : ${is_develop}"

if [[ ${is_develop} == '1' ]]; then
    ${py_bin} setup.py develop
    log "Installing to local python env done"
else
    rm -f ./dist/*.whl
    ${py_bin} setup.py build_ext -j $(($(nproc) - 1)) bdist_wheel

    whl=`find ./dist -name "*.whl"`
    if [[ -z ${whl} ]]; then
        echo "No .whl found after setup.py!"
        exit -1
    else
        log "Found the .whl: `realpath ${whl}`"
    fi
fi

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



function echoerr() {
    echo "$@" 1>&2;
}

download_dir=$1
pb_ver=$2

download_dir="${download_dir}/.download_cache"
mkdir -p ${download_dir}

protoc_dir="protoc-${pb_ver}-linux-x86_64"
protoc_file="${protoc_dir}.zip"
protoc_url="http://gitlab.alibaba-inc.com/odps_tensorflow/other/raw/master/github.com/protocolbuffers/protobuf/releases/download/v${pb_ver}/${protoc_file}"
protoc_url_bak="https://blade-disc-remote-cache.oss-cn-hongkong.aliyuncs.com/download/protobuf/v${pb_ver}/${protoc_file}"

if [ ! -f ${download_dir}/${protoc_file} ]; then
    code=`curl -s -L  -o ${download_dir}/${protoc_file} ${protoc_url} --write-out "%{http_code}"`
    if [[ "${code}" != "200" ]]; then
        code_bak=`curl -s -L  -o ${download_dir}/${protoc_file} ${protoc_url_bak} --write-out "%{http_code}"`
        if [[ "${code_bak}" != "200" ]]; then
            echoerr "Download failed:"
            echoerr "    Attempt 1: [${code}] ${protoc_url}"
            echoerr "    Attempt 2: [${code_bak}] ${protoc_url_bak}"
            exit -1
        fi
    fi
fi
rm -fr ${download_dir}/${protoc_dir}
unzip -q ${download_dir}/${protoc_file} -d ${download_dir}/${protoc_dir}

if [ ! -f ${download_dir}/${protoc_dir}/bin/protoc ]; then
    echoerr "protoc not found after downloading. Please delete ${download_dir}/${protoc_file} and retry."
    exit -1
fi

echo -n ${download_dir}/${protoc_dir}/bin/protoc

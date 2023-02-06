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

function install_llvm_toolchain() {
    llvm_version=clang+llvm-10.0.1-x86_64-linux-gnu-ubuntu-16.04
    llvm_tgz=${llvm_version}".tar.xz"
    local_tgz="/install/${llvm_tgz}"
    llvm_url="https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.1/${llvm_tgz}"

    echo "Download url: ${llvm_url}"

    wget -nv ${llvm_url} -O ${local_tgz}
    if [[ "$?" != "0" ]]; then
        echo "Download clang+llvm-12.0.1 failed."
        exit -1
    fi

    tar -xf ${local_tgz} --skip-old-files -C /usr/local
    mv /usr/local/${llvm_version} /usr/local/llvm_toolchain
    rm -f ${local_tgz}
}

get_arch=`arch`
if [[ $get_arch =~ "x86_64" ]]; then
    install_llvm_toolchain
else
    exit 0
fi

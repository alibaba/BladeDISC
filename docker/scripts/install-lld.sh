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
version=12.0.1
llvm_version=clang+llvm-12.0.1-x86_64-linux-gnu-ubuntu-16.04
llvm_tgz=${llvm_version}".tar.xz"
local_tgz="/install/${llvm_tgz}"
llvm_url="https://github.com/llvm/llvm-project/releases/download/llvmorg-${version}/${llvm_tgz}"

echo "Download url: ${llvm_url}"

wget -nv ${llvm_url} -O ${local_tgz}
if [[ "$?" != "0" ]]; then
    echo "Download clang+llvm-${version} failed."
    exit -1
fi

tar -xf ${local_tgz} --skip-old-files -C /usr/local
mv /usr/local/${llvm_version} /usr/local/llvm_toolchain
rm -f ${local_tgz}
<<'COMMENT'
wget -v https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.1-rc4/llvm-project-10.0.1rc4.tar.xz
tar -xf llvm-project-10.0.1rc4.tar.xz
rm -rf llvm-project && mv llvm-project-10.0.1rc4 llvm-project
rm -rf build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_PROJECTS=lld \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DLLVM_TARGETS_TO_BUILD="X86" \
      ../llvm-project/llvm
make install -j64
rm -rf build
COMMENT

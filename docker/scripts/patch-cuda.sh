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

# This script copies headers and libraries of cuDNN from /usr to /usr/local/cuda in
# nvidia:cuda-10.x images. This is needed because tensorflow assumes that.

device=$1
echo "Patching ${device} ..."

# Copy cudnn headers and lib for cuda 
headers=$(ls /usr/include/*.h | grep -E 'cublas|cudnn|nvblas')
for hdr in ${headers[@]}; do
    echo "Copy ${hdr} to cuda home."
    cp ${hdr} /usr/local/cuda/include/
done

libs=$(ls /usr/lib/x86_64-linux-gnu/ | grep -E 'cublas|cudnn|nvblas' | grep -E '\.so|\.a')
for lib in ${libs[@]}; do
    echo "Copy /usr/lib/x86_64-linux-gnu/${lib} to cuda home."
    cp /usr/lib/x86_64-linux-gnu/${lib} /usr/local/cuda/lib64/
done


if [[ ${device} == "cu102" ]]; then 
    echo "Installing cuda 10.2 patches ..."
    # patch 1
    curl -sL https://developer.download.nvidia.com/compute/cuda/10.2/Prod/patches/1/cuda_10.2.1_linux.run -o cuda_10.2.1_linux.run
    bash ./cuda_10.2.1_linux.run --silent --toolkit
    rm -f ./cuda_10.2.1_linux.run

    # patch 2
    curl -sL https://developer.download.nvidia.com/compute/cuda/10.2/Prod/patches/2/cuda_10.2.2_linux.run -o cuda_10.2.2_linux.run
    bash ./cuda_10.2.2_linux.run --silent --toolkit
    rm -f ./cuda_10.2.2_linux.run
fi
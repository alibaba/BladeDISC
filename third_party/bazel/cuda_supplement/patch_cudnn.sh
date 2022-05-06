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

set -e

[[ $# -ne 2 ]] && { echo "Usage: $0 <cuda_home> <out_dir>"; exit -1; }

cuda_home=$1
out_dir=$2

echo "Pathcing myelin libraries..."
echo "    CUDA Home        : ${cuda_home}"
echo "    Output Directory : ${out_dir}"

cudnn_lib="${cuda_home}/lib64/libcudnn_static.a"
patched_cudnn_lib="${out_dir}/libcudnn_static_patched.a"

objcopy --weaken-symbols $(dirname $0)/conflict_symbols.txt ${cudnn_lib} ${patched_cudnn_lib}

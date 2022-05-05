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

[[ $# -ne 2 ]] && { echo "Usage: $0 <trt_home> <out_dir>"; exit -1; }

trt_home=$1
out_dir=$2

echo "Pathcing myelin libraries..."
echo "    TensorRT Home    : ${trt_home}"
echo "    Output Directory : ${out_dir}"

compiler_lib="${trt_home}/lib/libmyelin_compiler_static.a"
executor_lib="${trt_home}/lib/libmyelin_executor_static.a"
patched_executor_lib="${out_dir}/libmyelin_executor_static_patched.a"

nm ${compiler_lib} | awk '/ T / && /myelin/ {print $3}' | sort > compiler_symbols.txt
nm ${executor_lib} | awk '/ T / && /myelin/ {print $3}' | sort > executor_symbols.txt

comm -12 compiler_symbols.txt executor_symbols.txt > common_symbols.txt
objcopy --weaken-symbols common_symbols.txt ${executor_lib} ${patched_executor_lib}

num_weakened=$(wc -l common_symbols.txt | awk '{print $1}')
echo "Patching done, ${num_weakened} symbols were weakened from ${executor_lib}."

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
#export TORCH_BLADE_MHLO_DEBUG_LOG=on
export DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL=true
export TORCH_MHLO_OP_WHITE_LIST="aten::clone;aten::var;aten::rsub;aten::amax;aten::to;aten::tanh;aten::_to_copy;prims::broadcast_in_dim;aten::new_zeros;aten::zeros_like;aten::select_scatter;aten::slice_scatter;aten::full_like;aten::where"
#python3 test_bert.py --backend aot_disc --batch-size 16 2>&1 | tee disc.b16.compare.log
#python3 test_bert.py --backend aot_disc --batch-size 32 2>&1 | tee disc.b32.compare.log
#python3 test_bert.py --backend aot_disc --batch-size 64 2>&1 | tee disc.b64.compare.log

# /opt/nvidia/nsight-systems/2022.4.1/bin/nsys nvprof --profile-from-start=off -o disc_debug.nvprof python3 test_bert.py --prof_dynamo --backend aot_disc_debug 2>&1 | tee disc_debug.nvprof.log
# /opt/nvidia/nsight-systems/2022.4.1/bin/nsys nvprof --profile-from-start=off -o disc.nvprof python3 test_bert.py --prof_dynamo --backend aot_disc 2>&1 | tee disc.nvprof.log
# /opt/nvidia/nsight-systems/2022.4.1/bin/nsys nvprof --profile-from-start=off -o eager.nvprof python3 test_bert.py --prof_baseline --backend aot_disc 2>&1 | tee eager.nvprof.log
function generate_prof() {
batch=$1
mode=$2
python3 test_bert.py --backend $mode --batch-size 8 2>&1 | tee b${batch}.${mode}.compare.log
# /opt/nvidia/nsight-systems/2022.4.1/bin/nsys nvprof --profile-from-start=off -o b${batch}.${mode}.nvprof \
#     python3 test_bert.py --prof_dynamo --backend ${mode} 2>&1 | tee b${batch}.${mode}.nvprof.log
# /opt/nvidia/nsight-systems/2022.4.1/bin/nsys profile -f true --wait=primary -c cudaProfilerApi -o b${batch}.${mode} \
#     python3 test_bert.py --batch-size $batch --prof_dynamo --backend ${mode} 2>&1 | tee b${batch}.${mode}.log
# rm -rf b${batch}.${mode}*
# /opt/nvidia/nsight-systems/2022.4.1/bin/nsys profile -o b${batch}.${mode} --stats false --force-overwrite true --kill=none --wait=primary -c cudaProfilerApi \
#      python3 test_bert.py --batch-size $batch --prof_dynamo --backend ${mode} 2>&1 | tee b${batch}.${mode}.log
# /opt/nvidia/nsight-systems/2022.4.1/bin/nsys stats --report gputrace -f column --force-overwrite -q -o b${batch}.${mode} b${batch}.${mode}.nsys-rep
# /opt/nvidia/nsight-systems/2022.4.1/bin/nsys stats --report gputrace -q -f csv --force-overwrite -o b${batch}.${mode} b${batch}.${mode}.nsys-rep
#  python3 parse_nsys_results.py b${batch}.${mode}_gputrace.csv > b${batch}.${mode}.report
# python3 parse_nsys_results.py --csv b${batch}.${mode}_gputrace.csv > b${batch}.${mode}.report.csv
}

function run_prof() {
batch=$1
generate_prof ${batch} aot_disc
# generate_prof ${batch} inductor
}

run_prof 8
# run_prof 16
# run_prof 32

# export TORCH_BLADE_DEBUG_LOG=on
#/opt/nvidia/nsight-systems/2022.4.1/bin/nsys profile -f true --wait=primary -c cudaProfilerApi -o disc python3 test_bert.py --prof_dynamo --backend aot_disc 2>&1 | tee disc.log
# /opt/nvidia/nsight-systems/2022.4.1/bin/nsys stats --report gputrace -f column --force-overwrite -q -o disc disc.nsys-rep
# /opt/nvidia/nsight-systems/2022.4.1/bin/nsys stats --report gputrace -q -f csv --force-overwrite -o disc disc.nsys-rep
# python3 parse_nsys_results.py disc_gputrace.csv > disc.report
# python3 parse_nsys_results.py disc_gputrace.csv --csv > disc.report.csv
# python3 parse_nsys_results.py --aggregate_kernel_with_prefix disc_gputrace.csv > disc.agg.report
# 
# /opt/nvidia/nsight-systems/2022.4.1/bin/nsys profile -f true --wait=primary -c cudaProfilerApi -o eager python3 test_bert.py --prof_baseline --backend aot_disc 2>&1 | tee eager.log
# /opt/nvidia/nsight-systems/2022.4.1/bin/nsys stats --report gputrace -f column --force-overwrite -q -o eager eager.nsys-rep
# /opt/nvidia/nsight-systems/2022.4.1/bin/nsys stats --report gputrace -q -f csv --force-overwrite -o eager eager.nsys-rep
# python3 parse_nsys_results.py eager_gputrace.csv > eager.report

# #/opt/nvidia/nsight-systems/2022.4.1/bin/nsys profile -f true --wait=primary -c cudaProfilerApi -o base python3 test_bert.py --prof_baseline --backend aot_disc 2>&1 | tee base.log
#/opt/nvidia/nsight-systems/2022.4.1/bin/nsys stats --report gputrace -q -f csv --force-overwrite -o base base.nsys-rep
#python3 parse_nsys_results.py base_gputrace.csv > base.report

# /opt/nvidia/nsight-systems/2022.4.1/bin/nsys stats --report gputrace -f column --force-overwrite -q -o base base.nsys-rep

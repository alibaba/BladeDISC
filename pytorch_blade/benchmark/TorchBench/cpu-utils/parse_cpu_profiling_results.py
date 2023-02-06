#!/usr/bin/python
# Copyright 2023 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# -*- coding: UTF-8 -*-
import sys
import re
import os
import numpy as np

disc_launch_counts = 0;
kernel_dict = {}
# examples:
#     [[DISC]] launch main_gKernel__229 elapsed : 5 us
#     [[DISC]] launch main_kLoop_convert__8_1_24_230 elapsed : 10 us
#     [[DISC]] launch ral_qconv_acl_s8_s8_s8_per_channel elapsed : 96 us
#     [[DISC]] launch main_kStitch_add_convert__42_2_23_231 elapsed : 36 us
def parse_one_line(line):
    if re.search("\[\[DISC\]\] launch .* elapsed : [0-9][0-9]* us", line):
        global disc_launch_counts
        disc_launch_counts += 1
        kernel = re.search(r'launch (.*?)(__[0-9].*)* elapsed : (.*?) us$', line)
        if kernel == None:
            print("kernel regex wrong")
            exit(1)
        kernel_name = kernel.group(1)
        assert len(kernel_name), 'expect kernel name'
        kernel_time = kernel.group(3)
        assert len(kernel_time), 'expect kernel time'
        #print(kernel_name, kernel_time)
        global kernel_dict
        if kernel_name not in kernel_dict:
            kernel_dict[kernel_name]=[]
        kernel_dict[kernel_name].append(int(kernel_time))

def print_message():
    global kernel_dict
    global disc_launch_counts
    print("===========================================")
    print(f"total launched kernels count = {disc_launch_counts}")
    print("===========================================")
    for kernel_name, kernel_times in sorted(kernel_dict.items()):
        print(f"counts: {len(kernel_times)}\tname: {kernel_name[-50:]}")
    print("===========================================")
    ## compute total time
    total_time=0
    for kernel_name, kernel_times in sorted(kernel_dict.items()):
        total_time += np.sum(kernel_times)
    for kernel_name, kernel_times in sorted(kernel_dict.items()):
        msg = (
            "counts: %d\tkernel name: %s\n"
            "    time(us) sum: %-7d\t ratio: %-2.2f%%\t avg: %-3.3f\t max: %-3d\t min: %-3d\t mid: %-3d\t"
            
        ) % (
            len(kernel_times),
            kernel_name,
            np.sum(kernel_times),
            np.sum(kernel_times) * 100 / total_time,
            np.mean(kernel_times),
            np.max(kernel_times),
            np.min(kernel_times),
            np.median(kernel_times)
        )
        print(msg)
    print("===========================================")
    print("Computation intensive kernels summary:")
    compute_intensive_total_counts=0
    compute_intensive_total_times=0
    for kernel_name, kernel_times in sorted(kernel_dict.items()):
        if (kernel_name.find("conv") != -1 or kernel_name.find("gemm") != -1 ):
            print("    {}: total counts = {}, avg time (ns) = {:.2f}".format(kernel_name, len(kernel_times), np.mean(kernel_times)))
            compute_intensive_total_counts+=len(kernel_times)
            compute_intensive_total_times+=np.sum(kernel_times)
    print("  Total launched counts: {}".format(compute_intensive_total_counts))
    print("  Total elapsed time (ms): {:.2f}".format(compute_intensive_total_times/1000))
    print("  AVG elapsed times (ns): {:.2f}".format(compute_intensive_total_times/compute_intensive_total_counts))

    print("===========================================")
    print("Summary")
    print(f"total launched kernels count = {disc_launch_counts}")
    print(f"total elapsed time = {total_time / 1000} ms")
    print("===========================================")

def main(argv):
  if len(argv) != 2:
    print('Usage:\n\tpython parse_profiling_result.py result_file_path\n')
    sys.exit(0)

  with open(argv[-1]) as fd:
    for l in fd:
      l = l.strip()
      if len(l) == 0: continue
      r = parse_one_line(l)
  print_message()

if __name__ == '__main__':
  main(sys.argv)


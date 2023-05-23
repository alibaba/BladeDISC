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

# !/bin/bash
if [ -f $HOME/.cache/proxy_config ]; then
  source $HOME/.cache/proxy_config
fi
script_dir=$(cd $(dirname "$0"); pwd)
benchmark_repo_dir=$HOME/.cache/torchbenchmark
disc_compile_cache_dir=$HOME/.cache/disc
rm -rf ${benchmark_repo_dir}
# cache benchmark repo
if [ ! -d $benchmark_repo_dir ]; then
    git clone -q https://github.com/pai-disc/torchbenchmark.git --recursive $benchmark_repo_dir
fi

# parse the arguments
HARDWARE=$1; shift  ## AArch64-yitian, AArch64-g6r, X86-intel, X86-amd
JOB=$1; shift       ## tiny, partial, full
VERSION=$1; shift   ## pre, 200

# compile cache
if [ -d $disc_compile_cache_dir ]; then
    rm -rf $disc_compile_cache_dir
fi
mkdir -p $disc_compile_cache_dir

# setup for torchbenchmark
pushd $benchmark_repo_dir
# cache venv in benchmark dir
if [[ $HARDWARE == AArch64* ]]; then
    rm -rf ./venv && cp -r /opt/venv_disc ./venv
    python3 -m virtualenv venv --system-site-packages && source venv/bin/activate
    python3 -m pip install -r $script_dir/cpu-utils/requirements_aarch64_${VERSION}.txt
else
    python3 -m virtualenv venv --system-site-packages && source venv/bin/activate
    python3 -m pip install -r $script_dir/cpu-utils/requirements_cpu_${VERSION}.txt
fi

git pull && git checkout main
git submodule update --init --recursive --depth 1
models_file=${script_dir}/cpu-utils/${JOB}_models.txt
while read line
do
    models+=("$line")
done < ${models_file}
echo ${models[@]}
python3 install.py ${models[@]} --continue_on_fail
pushd $script_dir # pytorch_blade/benchmark/TorchBench

# benchmark
date_str=$(date '+%Y%m%d-%H')
echo ${date_str}
oss_link=https://bladedisc-ci.oss-cn-hongkong.aliyuncs.com
oss_dir=oss://bladedisc-ci/TorchBench/cpu/${HARDWARE}/${JOB}/${date_str}-${VERSION}
OSSUTIL=ossutil
GH=gh
if [[ $HARDWARE == AArch64* ]]; then
    OSSUTIL=ossutil-arm64
    GH=gh_arm64
fi

# tiny job: one thread mode, one model, eager/disc/dynamo-disc
# partial job: one thread mode, full models, eager/disc/dynamo-disc
# full job: multi thread mode, full models, all possible backends
threads2cores=('2_0-1')
if [[ ${JOB} == 'full' ]]; then
    threads2cores=('1_0' \
                   '2_0-1' '2_0-3' \
                   '4_0-3' '4_0-7' \
                   '8_0-7' '8_0-15' \
                   '16_0-15' '16_0-31' \
                   '32_0-31' '32_0-63' \
                   '64_0-63')
fi
## generate config file for cpu
rm -rf $HOME/.cache/torchmark/CPU_${JOB}*.yaml
python3 cpu-utils/generate_yaml_for_cpu.py -j $JOB -p ${benchmark_repo_dir}

total_dir=${HARDWARE}.${JOB}.${date_str}.${VERSION}
rm -rf ${total_dir} && mkdir ${total_dir}
# setup benchmark env
export DISC_CPU_ENABLE_WEIGHT_PRE_PACKING=1
export DISC_ACL_HWCAP2=29695  ## only work on yitian
export TORCHBENCH_ATOL=1e-3 TORCHBENCH_RTOL=1e-3 TORCH_BLADE_ENABLE_COMPILATION_CACHE=true
if [[ $HARDWARE == "AArch64-yitian-amp" ]]; then
    # torchbenchmark using consin similarity when using lower precision
    # export TORCHBENCH_ATOL=1e-2 TORCHBENCH_RTOL=1e-2
    export DISC_CPU_ACL_USE_AMP=1
    export DNNL_DEFAULT_FPMATH_MODE=any
fi

for t2c in ${threads2cores[@]}
do
    TCtuple=($(echo ${t2c} | tr "_" "\n"))
    threads=${TCtuple[0]}
    cores=${TCtuple[1]}
    export OMP_NUM_THREADS=$threads
    core_binding=${HARDWARE}_threads_${threads}_cores_${cores}
    rm -rf ${core_binding} && rm -rf eval-cpu-fp32
    taskset -c ${cores} python3 ${benchmark_repo_dir}/.github/scripts/run-config.py \
            -c CPU_${JOB}.yaml -b ${benchmark_repo_dir} --output-dir .
    mv eval-cpu-fp32 ${core_binding} # rename
    cat ${core_binding}/summary.csv
    mv ${core_binding} ${total_dir}
done

python3 cpu-utils/parse_cpu_results.py -p ${total_dir}

tar -zcf ${script_dir}/${total_dir}.tar.gz ${total_dir}
/disc/scripts/ci/$OSSUTIL cp ${script_dir}/${total_dir}.tar.gz ${oss_dir}/
/disc/scripts/ci/$OSSUTIL cp -r ${script_dir}/${total_dir} ${oss_dir}/

if [ -d $disc_compile_cache_dir ]; then
    rm -rf $disc_compile_cache_dir
fi

popd # $benchmark_repo_dir
popd # BladeDISC/

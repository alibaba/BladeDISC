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
# cache benchmark repo
if [ ! -d $benchmark_repo_dir ]; then
    git clone -q https://github.com/pai-disc/torchbenchmark.git --recursive $benchmark_repo_dir
fi

# parse the arguments
HARDWARE=$1; shift  ## AArch64-yitian, AArch64-g6r, X86-intel, X86-amd
JOB=$1; shift       ## tiny, partial, full
FIELDS=("$@")

# setup for torchbenchmark
pushd $benchmark_repo_dir
# cache venv in benchmark dir
if [[ $HARDWARE == AArch64* ]]; then
    cp -r /opt/venv_disc ./venv
    python3 -m virtualenv venv --system-site-packages && source venv/bin/activate
    python3 -m pip install -q -r $script_dir/requirements_aarch64.txt
else
    python3 -m virtualenv venv --system-site-packages && source venv/bin/activate
    python3 -m pip install -q -r $script_dir/requirements_cpu.txt
fi

git pull && git checkout main
git submodule update --init --recursive --depth 1
python3 install.py --continue_on_fail

pushd $script_dir # pytorch_blade/benchmark/TorchBench
ln -s $benchmark_repo_dir torchbenchmark

# benchmark
# setup benchmark env
export DISC_CPU_ENABLE_WEIGHT_PRE_PACKING=1
if [[ $HAEDWARE == *yitian* ]];then
    echo DISC_ACL_HWCAP2=29695
    export DISC_ACL_HWCAP2=29695
fi
export TORCHBENCH_ATOL=1e-3 TORCHBENCH_RTOL=1e-3

bench_target=$JOB
config_file=blade_cpu_$JOB.yaml
results=()
threads2cores=('1_0' \
               '2_0-1' '2_0-3' \
               '4_0-3' '4_0-7' \
               '8_0-7' '8_0-15' \
               '16_0-15' '16_0-31' \
               '32_0-31' '32_0-63' \
               '64_0-63')
for _ in ${threads2cores[@]}
do
    TCtuple=($(echo $_ | tr "_" "\n"))
    threads=${TCtuple[0]}
    cores=${TCtuple[1]}
    result=eval_${HARDWARE}_fp32_threads_${threads}_cores_${cores}
    results[${#results[*]}]=$result
    export OMP_NUM_THREADS=$threads GOMP_CPU_AFFINITY=$cores
    taskset -c $cores python3 torchbenchmark/.github/scripts/run-config.py \
            -c $config_file -b ./torchbenchmark/ --output-dir .
    mv eval-cpu-fp32 $result
    if [[ $HAEDWARE == *yitian* ]];then
        ## enable amp for yitian
        echo DISC_CPU_ACL_USE_AMP=1, DNNL_DEFAULT_FPMATH_MODE=any
	export DISC_CPU_ACL_USE_AMP=1
        export DNNL_DEFAULT_FPMATH_MODE=any
	result=eval_${HARDWARE}_fp32_threads_${threads}_cores_${cores}_amp
	results[${#results[*]}]=$result
	taskset -c $cores python3 torchbenchmark/.github/scripts/run-config.py \
                -c $config_file -b ./torchbenchmark/ --output-dir .
        mv eval-cpu-fp32 $result
    fi
done

# results
date_str=$(date '+%Y%m%d-%H')
oss_link=https://bladedisc-ci.oss-cn-hongkong.aliyuncs.com
oss_dir=oss://bladedisc-ci/TorchBench/cpu/$HARDWARE/$JOB/${date_str}
OSSUTIL=ossutil
GH=gh
if [[ $HARDWARE == AArch64* ]]; then
    OSSUTIL=ossutil-arm64
    GH=gh_arm64
fi

for result in ${results[@]}
do
    cat ${result}/summary.csv
    curl ${oss_link}/TorchBench/baseline/${result}_${bench_target}.csv -o $result.csv
    tar -zcf ${script_dir}/${result}.tar.gz ${result}
    /disc/scripts/ci/$OSSUTIL cp  ${script_dir}/${result}.tar.gz ${oss_dir}/
    /disc/scripts/ci/$OSSUTIL cp -r ${script_dir}/${result} ${oss_dir}/${result}
done

# Default compare fields
if [ ${#FIELDS[@]} -eq 0 ]; then
    FIELDS=("disc (latency)" "dynamo-disc (latency)")
fi

# performance anaysis
python3 results_anaysis.py -t ${results} -i ${oss_dir} -p ${RELATED_DIFF_PERCENT} -f "${FIELDS[@]}"

if [ -f "ISSUE.md" ]; then
    wget ${oss_link}/download/github/$GH -O gh && chmod +x ./gh && \
    ./gh issue create -F ISSUE.md \
    -t "[TorchBench] Performance Signal Detected" \
    -l Benchmark
fi

popd # $benchmark_repo_dir
popd # BladeDISC/

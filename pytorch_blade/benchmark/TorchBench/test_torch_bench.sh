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

# setup for torchbenchmark
# for CI git-lfs permission problems
pushd $benchmark_repo_dir
# cache venv in benchmark dir
if [ $1 == "aarch64" ]; then
    cp -r /opt/venv_disc ./venv
fi
python3 -m virtualenv venv --system-site-packages && source venv/bin/activate
python3 -m pip install -q -r $script_dir/requirements_$1.txt
# install dependencies
git pull && git checkout bladnn && git submodule update --init --recursive --depth 1 && python3 install.py --continue_on_fail
# fix pycocotools after install
python3 -m pip install -U numpy
pushd $script_dir # pytorch_blade/benchmark/TorchBench
ln -s $benchmark_repo_dir torchbenchmark

# benchmark
# setup benchmark env
export DISC_EXPERIMENTAL_SPECULATION_TLP_ENHANCE=true \
    DISC_CPU_LARGE_CONCAT_NUM_OPERANDS=4 DISC_CPU_ENABLE_EAGER_TRANSPOSE_FUSION=1 \
    TORCHBENCH_ATOL=1e-2 TORCHBENCH_RTOL=1e-2
export BLADE_AUTH_USE_COUNTING=1 BLADE_GEMM_TUNE_JIT=1

bench_target=$2

if [[ $1 == "cpu" ]] || [[ $1 == "aarch64" ]]
then
    binding_cores=0
    config_file=blade_cpu_$2.yaml
    results=()
    declare -A threads2cores
    threads2cores=([1]="0" [2]="0-1" [4]="0-3" [8]="0-7")
    for threads in $(echo ${!threads2cores[*]})
    do
        cores=${threads2cores[$threads]}
        result=eval-$1-fp32_$threads
        results[${#results[*]}]=$result
        export OMP_NUM_THREADS=$threads GOMP_CPU_AFFINITY=$cores
        taskset -c $cores python3 torchbenchmark/.github/scripts/run-config.py \
                -c $config_file -b ./torchbenchmark/ --output-dir .
	mv eval-cpu-fp32 $result
    done
else
    config_file=blade_cuda_$2.yaml
    results=(eval-cuda-fp32 eval-cuda-fp16)
    python3 torchbenchmark/.github/scripts/run-config.py \
            -c $config_file -b ./torchbenchmark/ --output-dir .
fi

# results
date_str=$(date '+%Y%m%d-%H')
oss_link=https://bladedisc-ci.oss-cn-hongkong.aliyuncs.com
oss_dir=oss://bladedisc-ci/TorchBench/${bench_target}/${date_str}

for result in ${results[@]}
do
    cat ${result}/summary.csv
    curl ${oss_link}/TorchBench/baseline/${result}_${bench_target}.csv -o $result.csv
    if [ $1 == "aarch64" ]; then
        /disc/scripts/ci/ossutil-arm64 cp -r ${script_dir}/${result} ${oss_dir}/${result}
    else
        /disc/scripts/ci/ossutil cp -r ${script_dir}/${result} ${oss_dir}/${result}
    fi
done

# performance anaysis
python3 results_anaysis.py -t ${results} -i ${oss_dir} -p ${RELATED_DIFF_PERCENT}
if [ -f "ISSUE.md" ]; then
    wget ${oss_link}/download/github/gh && chmod +x ./gh && \
    ./gh issue create -F ISSUE.md \
    -t "[TorchBench] Performance Signal Detected" \
    -l Benchmark
fi

popd # $benchmark_repo_dir
popd # BladeDISC/

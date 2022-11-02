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
python3 -m virtualenv venv --system-site-packages && source venv/bin/activate
python3 -m pip install -q -r $script_dir/requirements_$1.txt
# install dependencies
git pull && git submodule update --init --recursive --depth 1 && python3 install.py --continue_on_fail
# fix pycocotools after install
python3 -m pip install -U numpy
pushd $script_dir # pytorch_blade/benchmark/TorchBench
ln -s $benchmark_repo_dir torchbenchmark

# benchmark 
config_file=blade_$1_$2.yaml
bench_target=$2
# setup benchmark env
export DISC_EXPERIMENTAL_SPECULATION_TLP_ENHANCE=true \
    DISC_CPU_LARGE_CONCAT_NUM_OPERANDS=4 DISC_CPU_ENABLE_EAGER_TRANSPOSE_FUSION=1 \
    OMP_NUM_THREADS=1 TORCHBENCH_ATOL=1e-2 TORCHBENCH_RTOL=1e-2
if [ $2 == "cpu" ];then
    # 4 cores
    export GOMP_CPU_AFFINITY="1-4"
fi
python3 torchbenchmark/.github/scripts/run-config.py -c $config_file -b ./torchbenchmark/ --output-dir .
# results



date_str=$(date '+%Y%m%d-%H')
if [ $2 == "cuda" ]
then
    cat eval-cuda-fp16/summary.csv
    cat eval-cuda-fp32/summary.csv
    /disc/scripts/ci/ossutil cp -r ${script_dir}/eval-cuda-fp16 oss://bladedisc-ci/TorchBench/${bench_target}/${date_str}/eval-cuda-fp16
    /disc/scripts/ci/ossutil cp -r ${script_dir}/eval-cuda-fp32 oss://bladedisc-ci/TorchBench/${bench_target}/${date_str}/eval-cuda-fp32
else
    cat eval-cpu-fp32/summary.csv
    /disc/scripts/ci/ossutil cp -r ${script_dir}/eval-cpu-fp32 oss://bladedisc-ci/TorchBench/${bench_target}/${date_str}/eval-cpu-fp32
fi

popd # $benchmark_repo_dir
popd # BladeDISC/
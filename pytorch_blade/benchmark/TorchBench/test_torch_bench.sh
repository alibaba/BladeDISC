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
# install dependencies
python3 -m virtualenv venv --system-site-packages && source venv/bin/activate
script_dir=$(cd $(dirname "$0"); pwd)
python3 -m pip install -q -r $script_dir/requirements.txt

# setup for torchbenchmark
benchmark_repo_dir=$HOME/.cache/torchbenchmark
if [ -d $benchmark_repo_dir ]; then
    rm -rf $benchmark_repo_dir
fi
git clone -q https://github.com/pai-disc/torchbenchmark.git --recursive $benchmark_repo_dir
# CI git-lfs permission problems
cd $benchmark_repo_dir && export HOME=$(pwd) && git lfs install --force
git pull && git submodule update --init --recursive --depth 1 && python3 install.py --continue_on_fail
pushd $script_dir # pytorch_blade/benchmark/TorchBench
# setup for torchdynamo
ln -s $benchmark_repo_dir torchbenchmark

# torchscript/torchdynamo frontend and disc backend
TORCHBENCH_ATOL=1e-2 TORCHBENCH_RTOL=1e-2 python3 torchbenchmark/.github/scripts/run-config.py -c blade_bench.yaml -b ./torchbenchmark/ --output-dir .
# results
cat eval-cuda-fp32/summary.csv
popd

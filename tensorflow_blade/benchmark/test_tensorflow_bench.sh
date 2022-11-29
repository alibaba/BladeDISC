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
benchmark_mdoel_dir=$HOME/.cache/TFBenchmark

date_str=$(date '+%Y%m%d-%H')
oss_dir=oss://bladedisc-ci/TFBench/${date_str}
result=eval-cuda-fp32.csv
pushd ${script_dir}

pip install -r requirement.txt
python3 benchmark.py -c model_config.yaml -p ${benchmark_mdoel_dir} -r ${result}

/disc/scripts/ci/ossutil cp -r ${script_dir}/${result} ${oss_dir}/${result}
popd

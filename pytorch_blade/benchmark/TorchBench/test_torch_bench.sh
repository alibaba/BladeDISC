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
apt install -y git git-lfs libglib2.0-0 libsndfile1 && pip install librosa torchdynamo torchvision torchaudio torchtext --extra-index-url https://download.pytorch.org/whl/cu113

script_dir=$(cd $(dirname "$0"); pwd)
pushd $script_dir
# setup for torchbenchmark
git clone https://github.com/jansel/benchmark.git --recursive torchbenchmark
cd torchbenchmark && python install.py && cd ../
# modified torchdynamo backend
cp backends.py /usr/local/lib/python3.8/dist-packages/torchdynamo/optimizations/

# dynamo frontend and disc backend
python torchbench.py --speedup-blade-disc -d cuda --isolate --float32 --skip-accuracy-check 2>&1 | tee speedup_blade.log

popd
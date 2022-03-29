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

oldpwd=$(pwd)
cwd=$(cd $(dirname "$0"); pwd)
cd $cwd
echo DIR: $(pwd)

wget -cnv https://bladedisc-ci.oss-cn-hongkong.aliyuncs.com/download/torch-blade/benchmark/torch-tensorrt/models.tar.gz -O models.tar.gz
tar xfz models.tar.gz

python3 perf_run.py --config=config/vgg16.yml
python3 perf_run.py --config=config/yolov5.yml
python3 perf_run.py --config=config/crnn.yml

cd $oldpwd

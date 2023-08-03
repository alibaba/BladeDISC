#!/bin/bash
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
set -ex

tensorrt_pkg=TensorRT-8.5.1.7.Linux.x86_64-gnu.cuda-11.8.cudnn8.6.tar.gz
curl -sL https://pai-blade.oss-accelerate.aliyuncs.com/build_deps/tensorrt/${tensorrt_pkg} -o ${tensorrt_pkg}

tar xvfz $tensorrt_pkg -C /usr/local/ 1>/dev/null 2>&1
ln -s /usr/local/${tensorrt_pkg%%.Linux*.tar.gz} /usr/local/TensorRT
rm -rf $tensorrt_pkg

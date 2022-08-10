#!/bin/bash
# Copyright 2021 The BladeDISC Authors. All rights reserved.
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
CUDA_RUN=/install/cuda_10.0.130_410.48_linux.run
wget -nv "https://pai-blade.oss-accelerate.aliyuncs.com/build_deps/cuda/cuda_10.0.130_410.48_linux.run" -O ${CUDA_RUN}
sudo sh ${CUDA_RUN} --silent --toolkit
rm -f ${CUDA_RUN}

CUDNN_TGZ=/install/cudnn-10.0-linux-x64-v7.6.4.38.tgz
wget -nv "https://pai-blade.oss-accelerate.aliyuncs.com/build_deps/cudnn/cudnn-10.0-linux-x64-v7.6.4.38.tgz" -O ${CUDNN_TGZ}
tar -xzf ${CUDNN_TGZ} --skip-old-files -C /usr/local/
rm -f ${CUDNN_TGZ}

rm -fr /usr/local/cuda-10.0/NsightCompute-2019.1
rm -fr /usr/local/cuda-10.0/nsightee_plugins
rm -fr /usr/local/cuda-10.0/extras
rm -fr /usr/local/cuda-10.0/libnsight
rm -fr /usr/local/cuda-10.0/libnvvp
rm -fr /usr/local/cuda-10.0/doc
rm -fr /usr/local/cuda-10.0/NsightSystems-2018.3
find /usr/local/cuda-10.0  -name "*.a" -delete

# cd /usr/local/cuda-10.0/targets/x86_64-linux/lib/stubs && ln -s libnvidia-ml.so libnvidia-ml.so.1

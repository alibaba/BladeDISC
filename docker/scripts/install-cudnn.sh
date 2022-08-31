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

# This script copies headers and libraries of cuDNN from /usr to /usr/local/cuda in
# nvidia:cuda-10.x images. This is needed because tensorflow assumes that.

cudnn_tgz=cudnn-11.4-linux-x64-v8.2.4.15.tgz
local_tgz="/install/${cudnn_tgz}"
cudnn_url="https://pai-blade.oss-accelerate.aliyuncs.com/build_deps/cudnn/${cudnn_tgz}"

echo "Download url: ${cudnn_url}"

wget -nv ${cudnn_url} -O ${local_tgz}
if [[ "$?" != "0" ]]; then
    echo "Download cuDNN failed."
    exit -1
fi

tar -xzf ${local_tgz} --skip-old-files -C /usr/local/
rm -f ${local_tgz}

# delete potential old version.
find /usr/lib/ -name libcudnn* -delete

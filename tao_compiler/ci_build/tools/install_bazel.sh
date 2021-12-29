#!/usr/bin/env bash
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



wget http://hlomodule.oss-cn-zhangjiakou.aliyuncs.com/tao_compiler/tools/bazelisk/v1.7.5/bazelisk-linux-amd64 -O /usr/local/bin/bazel
chmod +x /usr/local/bin/bazel

# This is a workaround for GFW.
# python3 packages
pip3 install numpy oss2 filelock
# replace system git with git wrapper
sys_git=$(which git)
if [ ! -f ${sys_git}.orig ];then
    mv $sys_git ${sys_git}.orig
    cp platform_alibaba/ci_build/install/git_wrapper.py $sys_git
fi

#!/usr/bin/env bash

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

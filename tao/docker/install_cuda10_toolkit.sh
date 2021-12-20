#!/bin/bash

set -ex
CUDA_RUN=/install/cuda_10.0.130_410.48_linux.run
wget -nv "http://hlomodule.oss-cn-zhangjiakou.aliyuncs.com/docker_deps/cuda_10.0.130_410.48_linux.run" -O ${CUDA_RUN}
sudo sh ${CUDA_RUN} --silent --toolkit
rm -f ${CUDA_RUN}

CUDNN_TGZ=/install/cudnn-10.0-linux-x64-v7.6.4.38.tgz
wget -nv "http://hlomodule.oss-cn-zhangjiakou.aliyuncs.com/docker_deps/cudnn-10.0-linux-x64-v7.6.4.38.tgz" -O ${CUDNN_TGZ}
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

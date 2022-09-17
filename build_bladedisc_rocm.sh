#!/bin/bash

set -euo pipefail

ENTRY=scripts/python/tao_build.py
VENV_PATH=/root/.venv/rocm
INSTALL_PATH=/usr/local
ROCM_ARGS="--rocm --rocm_path /opt/rocm-5.2.0"
BLADNN_ARGS="--platform_alibaba --blade_gemm"

export TF_DOWNLOAD_CLANG=0
export TF_ROCM_GCC=1

python ${ENTRY} ${VENV_PATH} -s configure --bridge-gcc default --compiler-gcc default $ROCM_ARGS $BLADNN_ARGS
python ${ENTRY} ${VENV_PATH} -s build_tao_bridge $ROCM_ARGS $BLADNN_ARGS
python ${ENTRY} ${VENV_PATH} -s build_tao_compiler $ROCM_ARGS $BLADNN_ARGS

cp tao/bazel-bin/libtao_ops.so ${INSTALL_PATH}/lib/
cp tf_community/bazel-out/k8-opt/bin/tensorflow/compiler/decoupling/tao_compiler_main ${INSTALL_PATH}/bin

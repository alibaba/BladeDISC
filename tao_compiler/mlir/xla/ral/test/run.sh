#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
export BRIDGE_ENABLE_TAO=true

export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_min_cluster_size=1"
export TF_DUMP_GRAPH_PREFIX="tmp/mlir"

export TAO_RAL_LIBRARY=/state/dev/xla/features/mlir_dhlo/tao_ral.so
export TAO_RAL_LOAD_FROM_SO_FILE

export TAO_ENABLE_RAL=true

#export TAO_RAL_LOAD_FROM_FILE="/state/dev/xla/features/mlir_dhlo/test/debug.so"

nvprof python codgen_kernels.py
nvprof python bace.py

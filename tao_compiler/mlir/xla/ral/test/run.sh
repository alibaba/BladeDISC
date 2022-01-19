#!/bin/sh
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

#!/bin/bash
# Copyright 2023 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#export TORCH_COMPILE_DEBUG=1
#export TORCH_COMPILE_DEBUG_DIR="debug"
#export AOT_FX_GRAPHS=true

export DISC_ENABLE_PREDEFINED_PDL=true
export DISC_TORCH_PDL_FILES="sd.pdll"
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="data"
export CLASS_DIR="class_data"
export OUTPUT_DIR="save-model"
export TORCH_BLADE_DEBUG_LOG=true
export TORCH_BLADE_ENABLE_COMPILATION_CACHE=true
export TORCH_BLADE_MHLO_DEBUG_LOG=true
export PYTHONPATH=diffusers/src:$PYTHONPATH
#export PYTHONPATH=/workspace/diffusers/src:$PYTHONPATH
export DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL=true
export BLADE_GEMM_TUNE_JIT=1
export TORCH_MHLO_OP_WHITE_LIST="aten::var;prims::broadcast_in_dim;aten::clone;aten::amax;aten::_to_copy;aten::floor;aten::upsample_nearest2d;aten::constant_pad_nd;aten::clamp;aten::slice_scatter;aten::new_zeros;aten::fill;aten::empty_like;aten::squeeze;aten::fill;aten::addcmul_;aten::addcmul_inplace_;aten::sqrt;aten::reciprocal;aten::addcdiv"
export DISC_GPU_ENABLE_TRANSPOSE_LIBRARY_CALL=true
export experimental_subgraph_conversion_parallelism=true
export TORCH_BLADE_EXPERIMENTAL_MERGE_HORIZONTAL_GROUPS=true
export BLADE_AUTH_USE_COUNTING=1
export DISC_ENABLE_HORIZONTAL_FUSION=true
export DISC_ENABLE_DOT_MERGE=false
ENTRY="nsys profile -f true -c cudaProfilerApi -o nsys-sd"
ENTRY=""
$ENTRY accelerate launch  --num_processes=1 train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --mixed_precision="fp16" \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of a man" \
  --class_prompt="a photo of man" \
  --resolution=512 \
  --train_batch_size=1 \
  --sample_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800 \
  --set_grads_to_none  2>&1 | tee train.log

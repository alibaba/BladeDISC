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


./triton-disc-opt tests/vec_add.mlir \
    --convert-triton-to-tritongpu \
    --convert-tritongpu-to-mlir-gpu  \
    --gpu-map-parallel-loops \
    --convert-parallel-loops-to-gpu \
    --gpu-kernel-outlining \
    --lower-affine \
    --disc-convert-gpu-to-nvvm \
    --disc-gpu-kernel-to-blob \
    --mlir-print-ir-after-all --dump-pass-pipeline

// Copyright 2021 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef AI_COMPILER_CUSTOM_LIB_DYN_SORT_H_
#define AI_COMPILER_CUSTOM_LIB_DYN_SORT_H_

#if GOOGLE_CUDA
#include <cuda_runtime.h>
using gpuStream_t = cudaStream_t;
#elif TENSORFLOW_USE_ROCM
#include <hip/hip_runtime.h>
using gpuStream_t = hipStream_t;
#endif

template <typename T>
struct LaunchTfTopKFunctor {
  void operator()(gpuStream_t stream, const T* input, int batch_size,
                  int length, int k, bool sorted, T* output, int* indices);
};

#endif  // AI_COMPILER_CUSTOM_LIB_DYN_SORT_H_

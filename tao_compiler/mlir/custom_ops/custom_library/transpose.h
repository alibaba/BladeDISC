// Copyright 2022 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef RAL_CUSTOM_LIBRARY_TRANSPOSE_H_
#define RAL_CUSTOM_LIBRARY_TRANSPOSE_H_

#include <vector>

#if GOOGLE_CUDA
#include <cuda_runtime.h>
using gpuStream_t = cudaStream_t;
#elif TENSORFLOW_USE_ROCM
#include <hip/hip_runtime.h>
using cudaError = int;
using gpuStream_t = hipStream_t;
#define cudaGetLastError hipGetLastError
#endif

namespace tao {
namespace ral {

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
template <typename T>
void LaunchTransposeKernel(gpuStream_t stream, T* input,
                           std::vector<int64_t> input_dims, T* output);
#endif
}  //  namespace ral
}  //  namespace tao

#endif
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

#ifndef RAL_CONTEXT_CUSTOM_LIBRARY_GPU_HELPER_
#define RAL_CONTEXT_CUSTOM_LIBRARY_GPU_HELPER_
#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "absl/types/variant.h"

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

template <typename... Ts, size_t... Is>
std::array<void*, sizeof...(Ts)> GetArrayOfElementPointersImpl(
    std::tuple<Ts...>* tuple, absl::index_sequence<Is...>) {
  return {{&std::get<Is>(*tuple)...}};
}
// Returns an array of void pointers to the elements of the given tuple.
template <typename... Ts>
std::array<void*, sizeof...(Ts)> GetArrayOfElementPointers(
    std::tuple<Ts...>* tuple) {
  return GetArrayOfElementPointersImpl(tuple,
                                       absl::index_sequence_for<Ts...>{});
}

// Launches a GPU kernel through cudaLaunchKernel in CUDA environment, or
// hipLaunchKernel in ROCm environment with the given arguments.
//
// The kernel parameters 'Ts' must be constructible from the arguments 'Args'.
template <typename... Ts, typename... Args>
void GpuLaunchKernel(void (*function)(Ts...), dim3 grid_dim, dim3 block_dim,
                     size_t shared_memory_size_bytes, gpuStream_t stream,
                     Args... arguments) {
  // static_assert(detail::NoneIsReference<Ts...>(),
  //              "Kernels with reference arguments have undefined behaviour.");
#if GOOGLE_CUDA
  auto func_ptr = absl::bit_cast<const void*>(function);
  // Cast arguments and forward them as an array of pointers.
  auto args_tuple = std::tuple<Ts...>(arguments...);
  auto arg_ptrs = GetArrayOfElementPointers(&args_tuple);
  auto result = cudaLaunchKernel(func_ptr, grid_dim, block_dim, arg_ptrs.data(),
                                 shared_memory_size_bytes, stream);
  // if (result != cudaSuccess) {
  //  return errors::Internal(cudaGetErrorString(result));
  //}
#elif TENSORFLOW_USE_ROCM
  hipLaunchKernelGGL(function, grid_dim, block_dim, shared_memory_size_bytes,
                     stream, std::forward<Args>(arguments)...);
#endif
}

}  //  namespace ral
}  //  namespace tao

#endif
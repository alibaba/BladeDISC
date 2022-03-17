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

// ------------------------------------------------------------------
// CUDA common used pre-defines
// ------------------------------------------------------------------

#ifndef __CUDA_COMMON_HEADER__
#define __CUDA_COMMON_HEADER__

#include <assert.h>

#include <iostream>
#if GOOGLE_CUDA
#include "cuda/include/cuda_runtime_api.h"
#else
#include <cuda_runtime_api.h>
#endif  // GOOGLE_CUDA

#define DIV_UP(a, b) (((a) + (b)-1) / (b))

static const int DEFAULT_1D_BLOCK_SIZE = 512;

static inline int GET_DEFAULT_1D_GRID_SIZE(const int N) {
  return DIV_UP(N, DEFAULT_1D_BLOCK_SIZE);
}

#define ASSERT(cond, msg)                                                  \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::cerr << "Assert `" #cond "` failed in " << __FILE__ << " line " \
                << __LINE__ << ": " << (msg) << std::endl;                 \
      std::terminate();                                                    \
    }                                                                      \
  } while (0)

#ifndef GOOGLE_CUDA

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#endif  // GOOGLE_CUDA

#define CUDA_CHECK(condition)                                                \
  /* Code block avoids redefinition of cudaError_t error */                  \
  do {                                                                       \
    cudaError_t error = (condition);                                         \
    if (error != cudaSuccess) {                                              \
      std::cerr << "CUDA error `" #condition "` in file " << __FILE__        \
                << " line " << __LINE__ << ": " << cudaGetErrorString(error) \
                << std::endl;                                                \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

#define LOC_DEV(ptr, byte_size)                                                \
  do {                                                                         \
    cudaError_t error = cudaMalloc((void**)(&(ptr)), (byte_size));             \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA Malloc Device error in file " << __FILE__ << " line " \
                << __LINE__ << ": " << cudaGetErrorString(error) << std::endl; \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define LOC_HST(ptr, byte_size)                                                \
  do {                                                                         \
    cudaError_t error = cudaMallocHost((void**)(&(ptr)), (byte_size));         \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA Malloc Host error in file " << __FILE__ << " line "   \
                << __LINE__ << ": " << cudaGetErrorString(error) << std::endl; \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define COPY_H2D(tag, src, byte_size)                                        \
  do {                                                                       \
    cudaError_t error =                                                      \
        cudaMemcpy((tag), (src), (byte_size), cudaMemcpyHostToDevice);       \
    if (error != cudaSuccess) {                                              \
      std::cerr << "CUDA Memcpy Host to Device error in file " << __FILE__   \
                << " line " << __LINE__ << ": " << cudaGetErrorString(error) \
                << std::endl;                                                \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

#define COPY_D2H(tag, src, byte_size)                                        \
  do {                                                                       \
    cudaError_t error =                                                      \
        cudaMemcpy((tag), (src), (byte_size), cudaMemcpyDeviceToHost);       \
    if (error != cudaSuccess) {                                              \
      std::cerr << "CUDA Memcpy Device to Host error in file " << __FILE__   \
                << " line " << __LINE__ << ": " << cudaGetErrorString(error) \
                << std::endl;                                                \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

#define COPY_D2D(tag, src, byte_size)                                        \
  do {                                                                       \
    cudaError_t error =                                                      \
        cudaMemcpy((tag), (src), (byte_size), cudaMemcpyDeviceToDevice);     \
    if (error != cudaSuccess) {                                              \
      std::cerr << "CUDA Memcpy Device to Device error in file " << __FILE__ \
                << " line " << __LINE__ << ": " << cudaGetErrorString(error) \
                << std::endl;                                                \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

#define COPY_H2D_ASYNC(tag, src, byte_size, stream)                        \
  do {                                                                     \
    cudaError_t error = cudaMemcpyAsync((tag), (src), (byte_size),         \
                                        cudaMemcpyHostToDevice, (stream)); \
    if (error != cudaSuccess) {                                            \
      std::cerr << "CUDA Memcpy Async Host to Device error in file "       \
                << __FILE__ << " line " << __LINE__ << ": "                \
                << cudaGetErrorString(error) << std::endl;                 \
      std::exit(EXIT_FAILURE);                                             \
    }                                                                      \
  } while (0)

#define COPY_D2H_ASYNC(tag, src, byte_size, stream)                        \
  do {                                                                     \
    cudaError_t error = cudaMemcpyAsync((tag), (src), (byte_size),         \
                                        cudaMemcpyDeviceToHost, (stream)); \
    if (error != cudaSuccess) {                                            \
      std::cerr << "CUDA Memcpy Async Device to Host error in file "       \
                << __FILE__ << " line " << __LINE__ << ": "                \
                << cudaGetErrorString(error) << std::endl;                 \
      std::exit(EXIT_FAILURE);                                             \
    }                                                                      \
  } while (0)

#define COPY_D2D_ASYNC(tag, src, byte_size, stream)                          \
  do {                                                                       \
    cudaError_t error = cudaMemcpyAsync((tag), (src), (byte_size),           \
                                        cudaMemcpyDeviceToDevice, (stream)); \
    if (error != cudaSuccess) {                                              \
      std::cerr << "CUDA Memcpy Async Device to Device error in file "       \
                << __FILE__ << " line " << __LINE__ << ": "                  \
                << cudaGetErrorString(error) << std::endl;                   \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

#define MEMSET_DEV(ptr, value, byte_size)                                      \
  do {                                                                         \
    cudaError_t error = cudaMemset((ptr), (value), (byte_size));               \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA Memset error in file " << __FILE__ << " line "        \
                << __LINE__ << ": " << cudaGetErrorString(error) << std::endl; \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#endif  // __CUDA_COMMON_HEADER__

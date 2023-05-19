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

#ifndef RAL_CUSTOM_LIBRARY_TF_TRANSPOSE_H_
#define RAL_CUSTOM_LIBRARY_TF_TRANSPOSE_H_

#include <cassert>
#include <vector>

#include "mlir/custom_ops/custom_library/dimensions.h"

#if GOOGLE_CUDA
#include <cuda_runtime.h>
using gpuStream_t = cudaStream_t;
#elif TENSORFLOW_USE_ROCM
#include <hip/hip_runtime.h>
using cudaError = int;
using gpuStream_t = hipStream_t;
#define cudaGetLastError hipGetLastError
#endif

#if !defined(__CUDACC__) && !defined(__HIPCC__)
dim3 threadIdx, blockDim, blockIdx;
#endif

namespace tao {
namespace ral {

// A helper function that converts a flat array index into a tensor index.
template <int IndexCount>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index<IndexCount> FlatToTensorIndex(
    int index, const Dimension<IndexCount>& dims) {
  Index<IndexCount> tensor_index;
  for (int i = IndexCount - 1; i >= 0; i--) {
    int new_index = index / dims[i];
    tensor_index[i] = index - dims[i] * new_index;
    index = new_index;
  }
  return tensor_index;
}

// A helper function that converts a tensor index into a flat array index.
template <int IndexCount>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int TensorIndexToFlat(
    const Index<IndexCount>& index, const Dimension<IndexCount>& dims) {
  int flat_index = index[0];
  for (int i = 1; i < IndexCount; i++) {
    flat_index = flat_index * dims[i] + index[i];
  }
  return flat_index;
}

// Use shared memory tiles to swap dimension-1 and dimension-2 of a 3D tensor,
// where dimensions are zero-based: output[i][j][k] = input[i][k][j].
//
// Each thread block operates on a single tile, a rectangle of dimensions
// TileSizeI x TileSizeJ.
//
// In general, for best performance, you should probably set TileSizeI,
// TileSizeJ equal to the number of threads in a warp (32 in nvidia GPUs).
// With a TileSizeI, TileSizeJ of 32, NumThreads of 128 or 256 seems to get
// the best performance on K40 GPUs.
template <typename T, int NumThreads, int TileSizeI, int TileSizeJ>
__global__ void SwapDimension1And2InTensor3UsingTiles(
    const T* __restrict__ input, Dimension<3> input_dims,
    T* __restrict__ output) {
  assert(blockDim.x == NumThreads);
  assert(blockDim.y == 1);
  assert(blockDim.z == 1);
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);

  constexpr int ReadRowPerPass = NumThreads / TileSizeJ;
  constexpr int WriteRowPerPass = NumThreads / TileSizeI;
  // One extra line in the inner dimension to avoid share memory bank conflict.
  // This is to mimic the following, but no constructor of T can be invoked.
  //     __shared__ T shared_memory_tile[TileSizeI][TileSizeJ + 1];
#if GOOGLE_CUDA
  __shared__ __align__(
      alignof(T)) char shared_mem_raw[TileSizeI * (TileSizeJ + 1) * sizeof(T)];
  typedef T(*SharedMemoryTile)[TileSizeJ + 1];
  SharedMemoryTile shared_memory_tile =
      reinterpret_cast<SharedMemoryTile>(shared_mem_raw);
#elif TENSORFLOW_USE_ROCM
  __shared__ T shared_memory_tile[TileSizeI][TileSizeJ + 1];
#endif

  int x = threadIdx.x;

  Dimension<3> output_dims = {
      input_dims[0],
      input_dims[2],
      input_dims[1],
  };

  Dimension<3> input_dims_in_tiles = {
      input_dims[0],
      (input_dims[1] + TileSizeI - 1) / TileSizeI,
      (input_dims[2] + TileSizeJ - 1) / TileSizeJ,
  };

  Index<3> input_tile_index =
      FlatToTensorIndex(blockIdx.x, input_dims_in_tiles);

  Index<3> input_tile_origin = {
      input_tile_index[0],
      input_tile_index[1] * TileSizeI,
      input_tile_index[2] * TileSizeJ,
  };

  int input_origin_flat_index =
      TensorIndexToFlat(input_tile_origin, input_dims);

  bool full_tile = true;
  int tile_width = TileSizeJ;

  // Only the last row or column may not have the full size.
  if (input_tile_index[2] == input_dims_in_tiles[2] - 1) {
    tile_width = input_dims[2] - (input_dims_in_tiles[2] - 1) * TileSizeJ;
    full_tile &= false;
  }

  int tile_height = TileSizeI;

  if (input_tile_index[1] == input_dims_in_tiles[1] - 1) {
    tile_height = input_dims[1] - (input_dims_in_tiles[1] - 1) * TileSizeI;
    full_tile &= false;
  }

  // Calculate effective thread number. This ensures that we use the largest
  // number of threads available to form a regular thread block with no
  // trailing incomplete lines.
  constexpr int in_effective_thread_num = NumThreads / TileSizeJ * TileSizeJ;

  if (x < in_effective_thread_num) {
    // Orient the logical thread block with respect to the input array.
    // ie. align the contiguous dimension of thread blocks with the contiguous
    // dimension of the input array.
    int ti = x / TileSizeJ;
    int tj = x % TileSizeJ;
    int input_index = input_origin_flat_index + ti * input_dims[2] + tj;
    int input_increment = ReadRowPerPass * input_dims[2];

    if (full_tile) {
#pragma unroll
      for (int i_loc = ti; i_loc < (TileSizeI); i_loc += ReadRowPerPass) {
        shared_memory_tile[i_loc][tj] = input[input_index];
        input_index += input_increment;
      }
    } else {
      if (tj < tile_width) {
        for (int i_loc = ti; i_loc < (tile_height); i_loc += ReadRowPerPass) {
          shared_memory_tile[i_loc][tj] = input[input_index];
          input_index += input_increment;
        }
      }
    }
  }

  __syncthreads();

  Index<3> output_tile_index = {
      input_tile_index[0],
      input_tile_index[2],
      input_tile_index[1],
  };

  Index<3> output_tile_origin = {
      output_tile_index[0],
      output_tile_index[1] * TileSizeJ,
      output_tile_index[2] * TileSizeI,
  };

  int output_origin_flat_index =
      TensorIndexToFlat(output_tile_origin, output_dims);

  constexpr int out_effective_thread_num = NumThreads / TileSizeI * TileSizeI;

  if (x < out_effective_thread_num) {
    // Re-orient the logical thread block with respect to the output array.
    // ie. align the contiguous dimension of thread blocks with contiguous
    // dimension of the output array.
    int ti = x / TileSizeI;
    int tj = x % TileSizeI;
    int output_index = output_origin_flat_index + ti * output_dims[2] + tj;
    int output_increment = WriteRowPerPass * output_dims[2];

    if (full_tile) {
#pragma unroll
      for (int i_loc = ti; i_loc < (TileSizeJ); i_loc += WriteRowPerPass) {
        output[output_index] = shared_memory_tile[tj][i_loc];
        output_index += output_increment;
      }
    } else {
      if (tj < tile_height) {
        for (int i_loc = ti; i_loc < (tile_width); i_loc += WriteRowPerPass) {
          output[output_index] = shared_memory_tile[tj][i_loc];
          output_index += output_increment;
        }
      }
    }
  }
}

}  //  namespace ral
}  //  namespace tao

#endif
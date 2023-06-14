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

#include "mlir/custom_ops/custom_library/gpu_helper.h"
#include "mlir/custom_ops/custom_library/tf_transpose.cu.h"
#include "mlir/custom_ops/custom_library/transpose.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace tao {
namespace ral {

template <typename IntegralType>
IntegralType CeilOfRatio(IntegralType numerator, IntegralType denominator) {
  const IntegralType rounded_toward_zero = numerator / denominator;
  const IntegralType intermediate_product = rounded_toward_zero * denominator;

  const bool needs_adjustment =
      (rounded_toward_zero >= 0) &&
      ((denominator > 0 && numerator > intermediate_product) ||
       (denominator < 0 && numerator < intermediate_product));
  const IntegralType adjustment = static_cast<IntegralType>(needs_adjustment);
  const IntegralType ceil_of_ratio = rounded_toward_zero + adjustment;
  return ceil_of_ratio;
}

#ifdef GOOGLE_CUDA
template <typename T>
void LaunchTransposeKernel(cudaStream_t stream, T* input,
                           std::vector<int64_t> input_dims, T* output) {
  static constexpr int64_t tile_size = 32;
  static constexpr int64_t num_threads = 256;
  Dimension<3> input_dims_in_tiles = {
      static_cast<int>(input_dims[0]),
      static_cast<int>(CeilOfRatio(input_dims[1], tile_size)),
      static_cast<int>(CeilOfRatio(input_dims[2], tile_size)),
  };
  Dimension<3> dims = {static_cast<int>(input_dims[0]),
                       static_cast<int>(input_dims[1]),
                       static_cast<int>(input_dims[2])};

  int total_tiles_count =
      input_dims_in_tiles[0] * input_dims_in_tiles[1] * input_dims_in_tiles[2];

  GpuLaunchKernel(SwapDimension1And2InTensor3UsingTiles<T, num_threads,
                                                        tile_size, tile_size>,
                  total_tiles_count, num_threads, 0, stream, input, dims,
                  output);
}
template void LaunchTransposeKernel<float>(cudaStream_t stream, float* input,
                                           std::vector<int64_t> input_dims,
                                           float* output);

template void LaunchTransposeKernel<Eigen::half>(
    cudaStream_t stream, Eigen::half* input, std::vector<int64_t> input_dims,
    Eigen::half* output);
#endif

}  //  namespace ral
}  //  namespace tao

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

#ifdef GOOGLE_CUDA
#include <cuda_runtime.h>
#endif

#include "mlir/custom_ops/custom_library/transpose.h"
#include "mlir/ral/context/context_util.h"
#include "mlir/ral/device/gpu/gpu_driver.h"
#include "mlir/ral/ral_context.h"
#include "mlir/ral/ral_driver.h"
#include "mlir/ral/ral_helper.h"
#include "mlir/ral/ral_logging.h"

// this file is required for Eigen::half
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
using tao::ral::buffer_t;
using tao::ral::Context;
using tao::ral::ExecutionContext;
using tao::ral::MemRefType;
using tao::ral::gpu::GPUDriver;
using tao::ral::gpu::stream_t;

namespace tao {
namespace ral {

DEFINE_TAO_TYPE_NAME_HELPER(Eigen::half, "f16");

#if defined(GOOGLE_CUDA) || \
    defined(TENSORFLOW_USE_ROCM) && !defined(TENSORFLOW_USE_DCU)

template <typename T, int N>
void ral_transpose(ExecutionContext* ctx, void* stream_handle,
                   MemRefType<T, N> input, MemRefType<int, 1> permute_value,
                   MemRefType<T, N> output) {
  T* d_in = input.data;
  T* d_out = output.data;

  static_assert(N == 2 || N == 3,
                "input of ral_transpose op should be rank2 or rank3 tensor");

  std::vector<int64_t> input_dims;
  if (N == 2)
    input_dims = {1, input.sizes[0], input.sizes[1]};
  else if (N == 3)
    input_dims = {input.sizes[0], input.sizes[1], input.sizes[2]};

  // get GPU driver, since we may need allocate extra temp gpu device workspace
  auto gpu_driver = ctx->getDriver<GPUDriver>(GPUDriver::name());
  auto stream =
      static_cast<gpuStream_t>(gpu_driver->asCUStream(ctx, stream_handle));

  LaunchTransposeKernel<T>(stream, d_in, input_dims, d_out);
}

TAO_RAL_API("ral_transpose", "gpu", ral_transpose<float, 2>);
TAO_RAL_API("ral_transpose", "gpu", ral_transpose<float, 3>);
TAO_RAL_API("ral_transpose", "gpu", ral_transpose<Eigen::half, 2>);
TAO_RAL_API("ral_transpose", "gpu", ral_transpose<Eigen::half, 3>);
#endif

}  //  namespace ral
}  //  namespace tao

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

#include "mlir/xla/ral/context/dynamic_sort_impl.h"

#include <vector>

#include "mlir/xla/ral/context/custom_library/dynamic_sort.h"
#include "mlir/xla/ral/device/gpu/gpu_driver.h"
#include "mlir/xla/ral/ral_driver.h"

using tao::ral::buffer_t;
using tao::ral::Context;
using tao::ral::ExecutionContext;
using tao::ral::MemRefType;
using tao::ral::gpu::GPUDriver;
using tao::ral::gpu::stream_t;

#if GOOGLE_CUDA
#include <cuda_runtime.h>
using gpuStream_t = cudaStream_t;
#elif TENSORFLOW_USE_ROCM
#include "rocm/include/hip/hip_runtime.h"
using gpuStream_t = hipStream_t;
#endif

namespace {

template <typename T>
void doTfTopK(ExecutionContext* ctx, GPUDriver* gpu_driver, void* stream_handle,
              const T* ikey, T* okey, int* oval, const SortDescriptor& desc,
              int64_t top_k) {
  auto stream =
      static_cast<gpuStream_t>(gpu_driver->asCUStream(ctx, stream_handle));
  LaunchTfTopKFunctor<T>()(stream, ikey, desc.batch, desc.sort_length, top_k,
                           true, okey, oval);
  return;
}

template <typename Tkey, typename Tval, unsigned int Rank = 1>
void ral_dsort(ExecutionContext* ctx, void* stream_handle,
               MemRefType<Tkey, Rank> keys, MemRefType<Tval, Rank> values,
               MemRefType<int32_t, 0> k, MemRefType<Tkey, Rank> out_keys,
               MemRefType<Tval, Rank> out_values, int64_t dimension,
               bool is_ascending) {
  // Note: do not need check keys and values shaped the same, which should be
  // done in previous hlo checking.
  auto keys_in = keys.data;
  auto values_in = values.data;
  auto keys_out = out_keys.data;
  auto values_out = out_values.data;
  auto top_k = *(k.data);
  if (top_k < -1) {
    ctx->signalError(Context::FAILURE, "Invalid ral_dsort with top k < -1");
    return;
  }
  auto desc = makeSortDescriptor(ctx, Rank, keys.sizes, is_ascending);
  if (top_k > desc.sort_length) {
    ctx->signalError(Context::FAILURE,
                     "Invalid Topk Sort with k > sorting length");
    return;
  }

  // get GPU driver, since we may need allocate extra temp gpu device workspace
  auto gpu_driver = ctx->getDriver<GPUDriver>(GPUDriver::name());

  doTfTopK(ctx, gpu_driver, stream_handle, keys_in, keys_out, values_out, desc,
           top_k);

  return;
}

}  // namespace

namespace tao {
namespace ral {

TAO_RAL_API("ral_dsort", "gpu", ral_dsort<float, int, 1>);
TAO_RAL_API("ral_dsort", "gpu", ral_dsort<int, int, 1>);
TAO_RAL_API("ral_dsort", "gpu", ral_dsort<float, int, 2>);
TAO_RAL_API("ral_dsort", "gpu", ral_dsort<int, int, 2>);

}  // namespace ral
}  // namespace tao

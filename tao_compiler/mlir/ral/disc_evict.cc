// Copyright 2024 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir/ral/context/base/cuda/cuda_context_impl.h"
#include "mlir/ral/context/pdll_util.h"
#include "mlir/ral/context/stream_executor_based_impl.h"
#include "mlir/ral/ral_helper.h"
#include "mlir/ral/context/context_util.h"
#include "mlir/ral/device/gpu/gpu_driver.h"
#include "mlir/ral/ral_context.h"
#include "mlir/ral/ral_driver.h"
#include "mlir/ral/ral_helper.h"

using tao::ral::Context;
using tao::ral::ExecutionContext;
using tao::ral::MemRefType;

namespace tao {
namespace ral {

template <typename T, int N>
MemRefType<bool, 0> ral_eviction_check(ExecutionContext* ctx, void* stream_handle,
                                MemRefType<T, N> input, void* customAttrs) {
  
  bool* data = (bool*)malloc(sizeof(bool));
  if(input.data == nullptr) {
    *data = true;
  } else {
    *data = false;
  }
  auto output = assignMemRef_0d<bool>(data);
  return output;
}

template <typename T, int N>
MemRefType<T, N> ral_evict(ExecutionContext* ctx, void* stream_handle,
                                    MemRefType<T, N> input, void* customAttrs) {
  auto evict_res = static_cast<gpu::BaseCudaExecutionContext*>(ctx)->eviction_manager.Evict(reinterpret_cast<int64_t>(input.data));
  if(evict_res) {
    static_cast<gpu::BaseCudaExecutionContext*>(ctx)->eviction_manager.TrackDealloc(input.data);
    auto output = assignMemRef<T, N>(nullptr, input.sizes);
    return output;
  }

  // Increase reference count
  auto it = static_cast<gpu::BaseCudaExecutionContext*>(ctx)->device_ptr_map.find(input.data);
  ++it->second;
  return input;
}


template <typename T, int N>
MemRefType<T, N> ral_shallow_copy(ExecutionContext* ctx, void* stream_handle,
                                    MemRefType<T, N> input, void* customAttrs) {
  // Increase reference count
  auto it = static_cast<gpu::BaseCudaExecutionContext*>(ctx)->device_ptr_map.find(input.data);
  ++it->second;
  return input;
}

TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<float, 1>);
TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<float, 2>);
TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<float, 3>);
TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<float, 4>);
TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<float, 5>);
TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<float, 6>);

TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<double, 1>);
TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<double, 2>);
TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<double, 3>);
TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<double, 4>);
TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<double, 5>);
TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<double, 6>);

TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<int32_t, 1>);
TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<int32_t, 2>);
TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<int32_t, 3>);
TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<int32_t, 4>);
TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<int32_t, 5>);
TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<int32_t, 6>);

TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<int64_t, 1>);
TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<int64_t, 2>);
TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<int64_t, 3>);
TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<int64_t, 4>);
TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<int64_t, 5>);
TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<int64_t, 6>);

TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<bool, 1>);
TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<bool, 2>);
TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<bool, 3>);
TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<bool, 4>);
TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<bool, 5>);
TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<bool, 6>);

TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<float16, 1>);
TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<float16, 2>);
TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<float16, 3>);
TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<float16, 4>);
TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<float16, 5>);
TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<float16, 6>);

TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<bfloat16, 1>);
TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<bfloat16, 2>);
TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<bfloat16, 3>);
TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<bfloat16, 4>);
TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<bfloat16, 5>);
TAO_RAL_API("ral_eviction_check", "cpu", ral_eviction_check<bfloat16, 6>);


TAO_RAL_API("ral_evict", "gpu", ral_evict<float, 1>);
TAO_RAL_API("ral_evict", "gpu", ral_evict<float, 2>);
TAO_RAL_API("ral_evict", "gpu", ral_evict<float, 3>);
TAO_RAL_API("ral_evict", "gpu", ral_evict<float, 4>);
TAO_RAL_API("ral_evict", "gpu", ral_evict<float, 5>);
TAO_RAL_API("ral_evict", "gpu", ral_evict<float, 6>);

TAO_RAL_API("ral_evict", "gpu", ral_evict<double, 1>);
TAO_RAL_API("ral_evict", "gpu", ral_evict<double, 2>);
TAO_RAL_API("ral_evict", "gpu", ral_evict<double, 3>);
TAO_RAL_API("ral_evict", "gpu", ral_evict<double, 4>);
TAO_RAL_API("ral_evict", "gpu", ral_evict<double, 5>);
TAO_RAL_API("ral_evict", "gpu", ral_evict<double, 6>);

TAO_RAL_API("ral_evict", "gpu", ral_evict<int32_t, 1>);
TAO_RAL_API("ral_evict", "gpu", ral_evict<int32_t, 2>);
TAO_RAL_API("ral_evict", "gpu", ral_evict<int32_t, 3>);
TAO_RAL_API("ral_evict", "gpu", ral_evict<int32_t, 4>);
TAO_RAL_API("ral_evict", "gpu", ral_evict<int32_t, 5>);
TAO_RAL_API("ral_evict", "gpu", ral_evict<int32_t, 6>);

TAO_RAL_API("ral_evict", "gpu", ral_evict<int64_t, 1>);
TAO_RAL_API("ral_evict", "gpu", ral_evict<int64_t, 2>);
TAO_RAL_API("ral_evict", "gpu", ral_evict<int64_t, 3>);
TAO_RAL_API("ral_evict", "gpu", ral_evict<int64_t, 4>);
TAO_RAL_API("ral_evict", "gpu", ral_evict<int64_t, 5>);
TAO_RAL_API("ral_evict", "gpu", ral_evict<int64_t, 6>);

TAO_RAL_API("ral_evict", "gpu", ral_evict<bool, 1>);
TAO_RAL_API("ral_evict", "gpu", ral_evict<bool, 2>);
TAO_RAL_API("ral_evict", "gpu", ral_evict<bool, 3>);
TAO_RAL_API("ral_evict", "gpu", ral_evict<bool, 4>);
TAO_RAL_API("ral_evict", "gpu", ral_evict<bool, 5>);
TAO_RAL_API("ral_evict", "gpu", ral_evict<bool, 6>);

TAO_RAL_API("ral_evict", "gpu", ral_evict<float16, 1>);
TAO_RAL_API("ral_evict", "gpu", ral_evict<float16, 2>);
TAO_RAL_API("ral_evict", "gpu", ral_evict<float16, 3>);
TAO_RAL_API("ral_evict", "gpu", ral_evict<float16, 4>);
TAO_RAL_API("ral_evict", "gpu", ral_evict<float16, 5>);
TAO_RAL_API("ral_evict", "gpu", ral_evict<float16, 6>);

TAO_RAL_API("ral_evict", "gpu", ral_evict<bfloat16, 1>);
TAO_RAL_API("ral_evict", "gpu", ral_evict<bfloat16, 2>);
TAO_RAL_API("ral_evict", "gpu", ral_evict<bfloat16, 3>);
TAO_RAL_API("ral_evict", "gpu", ral_evict<bfloat16, 4>);
TAO_RAL_API("ral_evict", "gpu", ral_evict<bfloat16, 5>);
TAO_RAL_API("ral_evict", "gpu", ral_evict<bfloat16, 6>);


TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<float, 1>);
TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<float, 2>);
TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<float, 3>);
TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<float, 4>);
TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<float, 5>);
TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<float, 6>);

TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<double, 1>);
TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<double, 2>);
TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<double, 3>);
TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<double, 4>);
TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<double, 5>);
TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<double, 6>);

TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<int32_t, 1>);
TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<int32_t, 2>);
TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<int32_t, 3>);
TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<int32_t, 4>);
TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<int32_t, 5>);
TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<int32_t, 6>);

TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<int64_t, 1>);
TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<int64_t, 2>);
TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<int64_t, 3>);
TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<int64_t, 4>);
TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<int64_t, 5>);
TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<int64_t, 6>);

TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<bool, 1>);
TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<bool, 2>);
TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<bool, 3>);
TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<bool, 4>);
TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<bool, 5>);
TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<bool, 6>);

TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<float16, 1>);
TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<float16, 2>);
TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<float16, 3>);
TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<float16, 4>);
TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<float16, 5>);
TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<float16, 6>);

TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<bfloat16, 1>);
TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<bfloat16, 2>);
TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<bfloat16, 3>);
TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<bfloat16, 4>);
TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<bfloat16, 5>);
TAO_RAL_API("ral_shallow_copy", "gpu", ral_shallow_copy<bfloat16, 6>);

}  //  namespace ral
}  //  namespace tao

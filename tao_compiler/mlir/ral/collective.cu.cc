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

#include "mlir/ral/collective.h"
#include "mlir/ral/context/base/cuda/cuda_context_impl.h"
#include "mlir/ral/context/pdll_util.h"
#include "mlir/ral/context/stream_executor_based_impl.h"
#include "mlir/ral/ral_helper.h"
#include "third_party/nccl/nccl.h"

using tao::ral::Context;
using tao::ral::ExecutionContext;
using tao::ral::MemRefType;

namespace tao {
namespace ral {

template <typename T>
struct ncclDataTypeMapper {};

template <>
struct ncclDataTypeMapper<float> {
  static const ncclDataType_t value = ncclFloat;
};

template <>
struct ncclDataTypeMapper<float16> {
  static const ncclDataType_t value = ncclHalf;
};

template <>
struct ncclDataTypeMapper<int> {
  static const ncclDataType_t value = ncclInt;
};

const std::unordered_map<std::string, ncclRedOp_t> kReductionTypeMap = {
    {"sum", ncclSum}, {"prod", ncclProd}, {"min", ncclMin}, {"max", ncclMax}};

ncclRedOp_t getNcclReductionType(const std::string& kind) {
  auto it = kReductionTypeMap.find(kind);
  if (it == kReductionTypeMap.end()) {
    return ncclSum;
  }
  return it->second;
}

template <typename T, int N>
MemRefType<T, N> ral_all_reduce(ExecutionContext* ctx, void* stream_handle,
                                MemRefType<T, N> input, void* customAttrs) {
  auto attr =
      getOrParsePDLAttr(ctx, customAttrs, "simple_test_fused_add_mul_kernel");
  if (!attr) {
    ctx->signalError(Context::FAILURE, "fail to parse custom_attrs\n");
  }
  auto& dictAttr = attr->as<DictPDLAttr>();
  std::string reductionKind =
      dictAttr.get("reduction_kind").template as<StrPDLAttr>().getValue();

  bool isAsync = dictAttr.get("is_async").template as<BoolPDLAttr>().getValue();

  ncclDataType_t ncclDtype = ncclDataTypeMapper<T>::value;
  auto ncclReductionType = getNcclReductionType(reductionKind);

  auto send_buffer = input.data;
  int element_count = 1;
  for (int i = 0; i < N; ++i) {
    element_count *= input.sizes[i];
  }
  auto gpu_driver = ctx->getDriver<tao::ral::gpu::GPUDriver>(
      tao::ral::gpu::GPUDriver::name());
  auto gpu_stream =
      static_cast<gpu::BaseCudaExecutionContext*>(ctx)->getCommStream();
  auto nccl_comm =
      static_cast<gpu::BaseCudaExecutionContext*>(ctx)->getNcclComm();
  auto ptr = static_cast<T*>(gpu_driver->alloc(ctx, element_count * sizeof(T)));
  auto output = assignMemRef<T, N>(ptr, input.sizes);
  auto recv_buffer = output.data;
  // TODO(yancey): support more nccl operations
  auto ncclResult =
      ncclAllReduce(send_buffer, recv_buffer, element_count, ncclDtype,
                    ncclReductionType, nccl_comm, gpu_stream);
  if (ncclResult != ncclSuccess) {
    ctx->signalError(Context::FAILURE, "fail to call ncclAllReduce\n");
  }

  if (isAsync && gpu_stream) {
    int64_t token_key =
        dictAttr.get("async_token_key").template as<IntPDLAttr>().getValue();
    cudaEvent_t event;

    auto event_status = cudaEventCreate(&event);
    if (event_status != cudaSuccess) {
      ctx->signalError(Context::FAILURE, "cudaEventCreate failed\n");
    }

    auto record_status = cudaEventRecord(event, gpu_stream);
    if (record_status != cudaSuccess) {
      cudaEventDestroy(event);
      ctx->signalError(Context::FAILURE, "cudaEventRecord failed\n");
    }

    static_cast<gpu::BaseCudaExecutionContext*>(ctx)->addAsyncPairToken(
        token_key, event);
  }

  return output;
}

template <typename T, int N>
MemRefType<T, N> ral_async_collective_done(ExecutionContext* ctx,
                                           void* stream_handle,
                                           MemRefType<T, N> input,
                                           void* customAttrs) {
  auto attr =
      getOrParsePDLAttr(ctx, customAttrs, "simple_test_fused_add_mul_kernel");
  if (!attr) {
    ctx->signalError(
        Context::FAILURE,
        "fail to parse custom_attrs in ral_async_collective_done\n");
  }

  auto& dictAttr = attr->as<DictPDLAttr>();
  int64_t token_key =
      dictAttr.get("async_token_key").template as<IntPDLAttr>().getValue();
  auto event =
      static_cast<gpu::BaseCudaExecutionContext*>(ctx)->getAsyncPairToken(
          token_key);
  if (event) {
    auto sync_status = cudaEventSynchronize(event);
    if (sync_status != cudaSuccess) {
      ctx->signalError(Context::FAILURE, "cudaEventSynchronize failed\n");
    }
    static_cast<gpu::BaseCudaExecutionContext*>(ctx)->removeAsyncPairToken(
        token_key);
    cudaEventDestroy(event);
  }

  // Increase ref count for input to prevent double free
  auto it =
      static_cast<gpu::BaseCudaExecutionContext*>(ctx)->device_ptr_map.find(
          input.data);
  ;
  ++it->second;

  return input;
}

TAO_RAL_API("ral_all_reduce", "gpu", ral_all_reduce<float, 1>);
TAO_RAL_API("ral_all_reduce", "gpu", ral_all_reduce<float, 2>);
TAO_RAL_API("ral_all_reduce", "gpu", ral_all_reduce<float, 3>);
TAO_RAL_API("ral_all_reduce", "gpu", ral_all_reduce<float, 4>);
TAO_RAL_API("ral_all_reduce", "gpu", ral_all_reduce<float16, 1>);
TAO_RAL_API("ral_all_reduce", "gpu", ral_all_reduce<float16, 2>);
TAO_RAL_API("ral_all_reduce", "gpu", ral_all_reduce<float16, 3>);
TAO_RAL_API("ral_all_reduce", "gpu", ral_all_reduce<float16, 4>);

TAO_RAL_API("ral_async_collective_done", "gpu",
            ral_async_collective_done<float, 1>);
TAO_RAL_API("ral_async_collective_done", "gpu",
            ral_async_collective_done<float, 2>);
TAO_RAL_API("ral_async_collective_done", "gpu",
            ral_async_collective_done<float, 3>);
TAO_RAL_API("ral_async_collective_done", "gpu",
            ral_async_collective_done<float, 4>);
TAO_RAL_API("ral_async_collective_done", "gpu",
            ral_async_collective_done<float16, 1>);
TAO_RAL_API("ral_async_collective_done", "gpu",
            ral_async_collective_done<float16, 2>);
TAO_RAL_API("ral_async_collective_done", "gpu",
            ral_async_collective_done<float16, 3>);
TAO_RAL_API("ral_async_collective_done", "gpu",
            ral_async_collective_done<float16, 4>);

}  //  namespace ral
}  //  namespace tao

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

template <typename T>
MemRefType<T, 0> ral_all_reduce_0d(ExecutionContext* ctx, void* stream_handle,
                                   MemRefType<T, 0> input, void* customAttrs) {
  auto attr = getOrParsePDLAttr(ctx, customAttrs, "ral_all_reduce");
  if (!attr) {
    ctx->signalError(Context::FAILURE, "fail to parse custom_attrs\n");
  }
  auto& dictAttr = attr->as<DictPDLAttr>();

  std::string reductionKind =
      dictAttr.get("reduction_kind").template as<StrPDLAttr>().getValue();
  ncclDataType_t ncclDtype = ncclDataTypeMapper<T>::value;
  auto ncclReductionType = getNcclReductionType(reductionKind);

  auto send_buffer = input.data;
  int input_elements = 1;
  auto gpu_driver = ctx->getDriver<tao::ral::gpu::GPUDriver>(
      tao::ral::gpu::GPUDriver::name());
  auto gpu_stream =
      static_cast<cudaStream_t>(gpu_driver->asCUStream(ctx, stream_handle));
  auto nccl_comm =
      static_cast<gpu::BaseCudaExecutionContext*>(ctx)->getNcclComm();
  auto ptr =
      static_cast<T*>(gpu_driver->alloc(ctx, input_elements * sizeof(T)));
  auto output = assignMemRef_0d<T>(ptr);
  auto recv_buffer = output.data;
  auto ncclResult =
      ncclAllReduce(send_buffer, recv_buffer, input_elements, ncclDtype,
                    ncclReductionType, nccl_comm, gpu_stream);
  if (ncclResult != ncclSuccess) {
    ctx->signalError(Context::FAILURE, "fail to call ncclAllReduce\n");
  }
  return output;
}

template <typename T, int N>
MemRefType<T, N> ral_all_reduce(ExecutionContext* ctx, void* stream_handle,
                                MemRefType<T, N> input, void* customAttrs) {
  auto attr = getOrParsePDLAttr(ctx, customAttrs, "ral_all_reduce");
  if (!attr) {
    ctx->signalError(Context::FAILURE, "fail to parse custom_attrs\n");
  }
  auto& dictAttr = attr->as<DictPDLAttr>();

  std::string reductionKind =
      dictAttr.get("reduction_kind").template as<StrPDLAttr>().getValue();
  ncclDataType_t ncclDtype = ncclDataTypeMapper<T>::value;
  auto ncclReductionType = getNcclReductionType(reductionKind);

  auto send_buffer = input.data;
  int input_elements = 1;
  for (int i = 0; i < N; ++i) {
    input_elements *= input.sizes[i];
  }
  auto gpu_driver = ctx->getDriver<tao::ral::gpu::GPUDriver>(
      tao::ral::gpu::GPUDriver::name());
  auto gpu_stream =
      static_cast<cudaStream_t>(gpu_driver->asCUStream(ctx, stream_handle));
  auto nccl_comm =
      static_cast<gpu::BaseCudaExecutionContext*>(ctx)->getNcclComm();
  auto ptr =
      static_cast<T*>(gpu_driver->alloc(ctx, input_elements * sizeof(T)));
  auto output = assignMemRef<T, N>(ptr, input.sizes);
  auto recv_buffer = output.data;
  auto ncclResult =
      ncclAllReduce(send_buffer, recv_buffer, input_elements, ncclDtype,
                    ncclReductionType, nccl_comm, gpu_stream);
  if (ncclResult != ncclSuccess) {
    ctx->signalError(Context::FAILURE, "fail to call ncclAllReduce\n");
  }
  return output;
}

template <typename T, int N>
MemRefType<T, N> ral_all_gather(ExecutionContext* ctx, void* stream_handle,
                                MemRefType<T, N> input, void* customAttrs) {
  T* send_buffer = input.data;
  auto attr = getOrParsePDLAttr(ctx, customAttrs, "ral_all_reduce");
  if (!attr) {
    ctx->signalError(Context::FAILURE, "fail to parse custom_attrs\n");
  }
  auto& dictAttr = attr->as<DictPDLAttr>();
  int all_gather_dim =
      dictAttr.get("all_gather_dim").template as<IntPDLAttr>().getValue();
  auto replic_groups =
      dictAttr.get("replica_groups").template as<DenseElementsPDLAttr>();
  int output_sizes[N];
  for (int i = 0; i < N; ++i) output_sizes[i] = input.sizes[i];
  output_sizes[all_gather_dim] =
      input.sizes[all_gather_dim] * replic_groups.getShape()[1];

  auto gpu_driver = ctx->getDriver<tao::ral::gpu::GPUDriver>(
      tao::ral::gpu::GPUDriver::name());
  auto gpu_stream =
      static_cast<cudaStream_t>(gpu_driver->asCUStream(ctx, stream_handle));
  auto nccl_comm =
      static_cast<gpu::BaseCudaExecutionContext*>(ctx)->getNcclComm();
  int input_elements = 1;
  for (int i = 0; i < N; ++i) {
    input_elements *= input.sizes[i];
  }
  int output_elements = input_elements * replic_groups.getShape()[1];
  auto ptr =
      static_cast<T*>(gpu_driver->alloc(ctx, output_elements * sizeof(T)));
  auto output = assignMemRef<T, N>(ptr, output_sizes);
  auto recv_buffer = output.data;

  ncclDataType_t ncclDtype = ncclDataTypeMapper<T>::value;

  if (ncclSuccess != ncclAllGather(send_buffer, recv_buffer, input_elements,
                                   ncclDtype, nccl_comm, gpu_stream)) {
    ctx->signalError(Context::FAILURE, "fail to call ncclAllGather\n");
  }
  return output;
}

template <typename T, int N>
MemRefType<T, N> ral_reduce_scatter(ExecutionContext* ctx, void* stream_handle,
                                    MemRefType<T, N> input, void* customAttrs) {
  T* send_buffer = input.data;
  auto attr = getOrParsePDLAttr(ctx, customAttrs, "ral_reduce_scatter");
  if (!attr) {
    ctx->signalError(Context::FAILURE, "fail to parse custom_attrs\n");
  }
  auto& dictAttr = attr->as<DictPDLAttr>();
  int scatter_dimension =
      dictAttr.get("scatter_dimension").template as<IntPDLAttr>().getValue();
  auto replic_groups =
      dictAttr.get("replica_groups").template as<DenseElementsPDLAttr>();
  std::string reductionKind =
      dictAttr.get("reduction_kind").template as<StrPDLAttr>().getValue();
  auto ncclReductionType = getNcclReductionType(reductionKind);

  int output_sizes[N];
  for (int i = 0; i < N; ++i) output_sizes[i] = input.sizes[i];
  output_sizes[scatter_dimension] =
      input.sizes[scatter_dimension] / replic_groups.getShape()[1];

  auto gpu_driver = ctx->getDriver<tao::ral::gpu::GPUDriver>(
      tao::ral::gpu::GPUDriver::name());
  auto gpu_stream =
      static_cast<cudaStream_t>(gpu_driver->asCUStream(ctx, stream_handle));
  auto nccl_comm =
      static_cast<gpu::BaseCudaExecutionContext*>(ctx)->getNcclComm();
  int output_elements = 1;
  for (int i = 0; i < N; ++i) {
    output_elements *= output_sizes[i];
  }
  auto ptr =
      static_cast<T*>(gpu_driver->alloc(ctx, output_elements * sizeof(T)));
  auto output = assignMemRef<T, N>(ptr, output_sizes);
  auto recv_buffer = output.data;

  ncclDataType_t ncclDtype = ncclDataTypeMapper<T>::value;

  if (ncclSuccess !=
      ncclReduceScatter(send_buffer, recv_buffer, output_elements, ncclDtype,
                        ncclReductionType, nccl_comm, gpu_stream)) {
    ctx->signalError(Context::FAILURE, "fail to call ncclReduceScatter\n");
  }
  return output;
}

TAO_RAL_API("ral_all_reduce", "gpu", ral_all_reduce_0d<float>);
TAO_RAL_API("ral_all_reduce", "gpu", ral_all_reduce<float, 1>);
TAO_RAL_API("ral_all_reduce", "gpu", ral_all_reduce<float, 2>);
TAO_RAL_API("ral_all_reduce", "gpu", ral_all_reduce<float, 3>);
TAO_RAL_API("ral_all_reduce", "gpu", ral_all_reduce<float, 4>);
TAO_RAL_API("ral_all_reduce", "gpu", ral_all_reduce_0d<float16>);
TAO_RAL_API("ral_all_reduce", "gpu", ral_all_reduce<float16, 1>);
TAO_RAL_API("ral_all_reduce", "gpu", ral_all_reduce<float16, 2>);
TAO_RAL_API("ral_all_reduce", "gpu", ral_all_reduce<float16, 3>);
TAO_RAL_API("ral_all_reduce", "gpu", ral_all_reduce<float16, 4>);

TAO_RAL_API("ral_all_gather", "gpu", ral_all_gather<float, 1>);
TAO_RAL_API("ral_all_gather", "gpu", ral_all_gather<float, 2>);
TAO_RAL_API("ral_all_gather", "gpu", ral_all_gather<float, 3>);
TAO_RAL_API("ral_all_gather", "gpu", ral_all_gather<float, 4>);
TAO_RAL_API("ral_all_gather", "gpu", ral_all_gather<float16, 1>);
TAO_RAL_API("ral_all_gather", "gpu", ral_all_gather<float16, 2>);
TAO_RAL_API("ral_all_gather", "gpu", ral_all_gather<float16, 3>);
TAO_RAL_API("ral_all_gather", "gpu", ral_all_gather<float16, 4>);

TAO_RAL_API("ral_reduce_scatter", "gpu", ral_reduce_scatter<float, 1>);
TAO_RAL_API("ral_reduce_scatter", "gpu", ral_reduce_scatter<float, 2>);
TAO_RAL_API("ral_reduce_scatter", "gpu", ral_reduce_scatter<float, 3>);
TAO_RAL_API("ral_reduce_scatter", "gpu", ral_reduce_scatter<float, 4>);
TAO_RAL_API("ral_reduce_scatter", "gpu", ral_reduce_scatter<float16, 1>);
TAO_RAL_API("ral_reduce_scatter", "gpu", ral_reduce_scatter<float16, 2>);
TAO_RAL_API("ral_reduce_scatter", "gpu", ral_reduce_scatter<float16, 3>);
TAO_RAL_API("ral_reduce_scatter", "gpu", ral_reduce_scatter<float16, 4>);

}  //  namespace ral
}  //  namespace tao

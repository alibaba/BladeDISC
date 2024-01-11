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
struct ncclDataTypeMapper<Eigen::half> {
  static const ncclDataType_t value = ncclHalf;
};

template <>
struct ncclDataTypeMapper<int> {
  static const ncclDataType_t value = ncclInt;
};

template <typename T, int N>
void all_reduce(ExecutionContext* ctx, void* stream_heandle,
                MemRefType<T, N> input, MemRefType<T, N> output) {
  ncclDataType_t ncclDtype = ncclDataTypeMapper<T>::value;
  auto send_buffer = input.data;
  auto recv_buffer = output.data;
  int element_count = 1;
  for (int i = 0; i < N; ++i) {
    element_count *= input.sizes[i];
  }
  auto gpu_driver = ctx->getDriver<tao::ral::gpu::GPUDriver>(
      tao::ral::gpu::GPUDriver::name());
  auto gpu_stream =
      static_cast<cudaStream_t>(gpu_driver->asCUStream(ctx, stream_heandle));
  auto nccl_comm =
      static_cast<gpu::BaseCudaExecutionContext*>(ctx)->getNcclComm();
  // TODO(yancey): support more nccl operations
  auto ncclResult = ncclAllReduce(send_buffer, recv_buffer, element_count,
                                  ncclDtype, ncclSum, nccl_comm, gpu_stream);
  if (ncclResult != ncclSuccess) {
    ctx->signalError(Context::FAILURE, "fail to call ncclAllReduce\n");
  }
}

TAO_RAL_API("all_reduce", "gpu", all_reduce<float, 1>);
TAO_RAL_API("all_reduce", "gpu", all_reduce<float, 2>);
TAO_RAL_API("all_reduce", "gpu", all_reduce<float, 3>);
TAO_RAL_API("all_reduce", "gpu", all_reduce<float, 4>);
TAO_RAL_API("all_reduce", "gpu", all_reduce<Eigen::half, 1>);
TAO_RAL_API("all_reduce", "gpu", all_reduce<Eigen::half, 2>);
TAO_RAL_API("all_reduce", "gpu", all_reduce<Eigen::half, 3>);
TAO_RAL_API("all_reduce", "gpu", all_reduce<Eigen::half, 4>);
}  //  namespace ral
}  //  namespace tao

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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_RAL_CONTEXT_STREAM_EXECUTOR_BASED_IMPL_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_RAL_CONTEXT_STREAM_EXECUTOR_BASED_IMPL_H_

#include "mlir/ral/ral_helper.h"

#ifdef TAO_RAL_USE_STREAM_EXECUTOR

#ifdef DISC_BUILD_FROM_TF_BRIDGE
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"
#include "tensorflow/stream_executor/stream_executor.h"
#else  // DISC_BUILD_FROM_TF_BRIDGE
#include "tensorflow/compiler/xla/stream_executor/device_memory.h"
#include "tensorflow/compiler/xla/stream_executor/multi_platform_manager.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#endif  // DISC_BUILD_FROM_TF_BRIDGE

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

#if defined(PLATFORM_ALIBABA) and defined(ENABLE_BLADE_GEMM)
#include "bladnn/bladnn.h"
#endif
namespace tao {
namespace ral {

DEFINE_TAO_TYPE_NAME_HELPER(Eigen::half, "f16");

namespace gpu {

#if defined(PLATFORM_ALIBABA) and defined(ENABLE_BLADE_GEMM)
template <typename T>
bladnn::Dtype toBlaDNNDtype() {
  if (std::is_same<T, int8_t>::value) {
    return bladnn::Dtype::kS8;
  }
  if (std::is_same<T, Eigen::half>::value) {
    return bladnn::Dtype::kF16;
  }
  if (std::is_same<T, float>::value) {
    return bladnn::Dtype::kF32;
  }
  if (std::is_same<T, double>::value) {
    return bladnn::Dtype::kF64;
  }
  return bladnn::Dtype::kUnknown;
}
#endif

}  // namespace gpu
}  // namespace ral
}  // namespace tao

#endif

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_RAL_CONTEXT_STREAM_EXECUTOR_BASED_IMPL_H_

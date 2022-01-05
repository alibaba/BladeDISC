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

#include "tensorflow/compiler/mlir/xla/ral/ral_helper.h"

#ifdef TAO_RAL_USE_STREAM_EXECUTOR

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"

namespace tao {
namespace ral {

DEFINE_TAO_TYPE_NAME_HELPER(Eigen::half, "f16");

namespace gpu {}  // namespace gpu
}  // namespace ral
}  // namespace tao

#endif

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_RAL_CONTEXT_STREAM_EXECUTOR_BASED_IMPL_H_

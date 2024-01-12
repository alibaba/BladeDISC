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

#ifndef TAO_COMPILER_MLIR_CUSTOM_OPS_COLLECTIVES_H_
#define TAO_COMPILER_MLIR_CUSTOM_OPS_COLLECTIVES_H_
#include "mlir/ral/context/context_util.h"
#include "mlir/ral/device/gpu/gpu_driver.h"
#include "mlir/ral/ral_context.h"
#include "mlir/ral/ral_driver.h"
#include "mlir/ral/ral_helper.h"
#include "third_party/nccl/nccl.h"

using tao::ral::ExecutionContext;
using tao::ral::MemRefType;
using tao::ral::gpu::GPUDriver;
using tao::ral::gpu::stream_t;
namespace tao {
namespace ral {

template <typename T, int N>
void all_reduce(ExecutionContext* ctx, void* stream_heandl,
                MemRefType<T, N> input, MemRefType<T, N> output);
}
}  // namespace tao
#endif
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

#ifndef RAL_CONTEXT_INIT_STREAM_EXECUTOR_H_
#define RAL_CONTEXT_INIT_STREAM_EXECUTOR_H_

#include "mlir/xla/ral/context/stream_executor_based_impl.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_types.h"
#include "tensorflow/compiler/xla/stream_executor/stream.h"

namespace tao {
namespace ral {
namespace gpu {

::stream_executor::Stream* GetOrCreateDefaultCudaStreamExecutorStream(
    int device_ordinal = 0,
    ::stream_executor::gpu::GpuStreamHandle stream = nullptr);

}  // namespace gpu
}  // namespace ral
}  // namespace tao

#endif  // RAL_CONTEXT_INIT_STREAM_EXECUTOR_H_

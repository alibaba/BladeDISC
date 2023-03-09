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

#include "mlir/xla/ral/context/base/cuda/cuda_stream.h"

namespace tao {
namespace ral {
namespace gpu {

namespace se = ::stream_executor;

bool CUDAStream::Init() {
  return se::gpu::GpuDriver::InitEvent(
             parent_->gpu_context(), &completed_event_,
             se::gpu::GpuDriver::EventFlags::kDisableTiming)
      .ok();
}

void CUDAStream::Destroy() {
  // it is not the duty of stream executor to destory the cuda stream
  if (completed_event_ != nullptr) {
    tsl::Status status = se::gpu::GpuDriver::DestroyEvent(
        parent_->gpu_context(), &completed_event_);
    if (!status.ok()) {
      LOG(ERROR) << status.error_message();
    }
  }
}

}  // namespace gpu
}  // namespace ral
}  // namespace tao

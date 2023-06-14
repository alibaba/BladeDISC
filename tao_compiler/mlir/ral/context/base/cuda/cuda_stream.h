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

#pragma once
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_types.h"

namespace tao {
namespace ral {
namespace gpu {

using ::stream_executor::gpu::GpuStreamHandle;

class CUDAStream : public ::stream_executor::gpu::GpuStream {
 public:
  explicit CUDAStream(::stream_executor::gpu::GpuExecutor* parent,
                      GpuStreamHandle gpu_stream)
      : ::stream_executor::gpu::GpuStream(parent) {
    gpu_stream_ = gpu_stream;
  }
  // Note: teardown is handled by a parent's call to DeallocateStream.
  ~CUDAStream() override {}
  bool Init() override;
  void Destroy() override;
};
}  // namespace gpu
}  // namespace ral
}  // namespace tao

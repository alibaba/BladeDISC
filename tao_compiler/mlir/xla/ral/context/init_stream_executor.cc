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

#include "tensorflow/compiler/mlir/xla/ral/context/init_stream_executor.h"

#include "tensorflow/compiler/mlir/xla/ral/context/base/cuda/cuda_stream.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/compiler/xla/stream_executor/stream.h"
#include "tensorflow/core/util/env_var.h"
#if TENSORFLOW_USE_ROCM
#include "tensorflow/compiler/xla/stream_executor/rocm/rocm_platform.h"
#else
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_platform.h"
#endif

namespace tao {
namespace ral {
namespace gpu {

namespace se = ::stream_executor;

struct TableEntry {
  se::Stream* stream;
  se::gpu::GpuContextHandle context;
  void ActivateContext() {
#if TENSORFLOW_USE_ROCM
    // hipContext is meaningless for now.
    auto errcode = hipCtxSetCurrent(context);
    assert(errcode == hipSuccess);
#else
    auto errcode = cuCtxSetCurrent(context);
    assert(errcode == CUDA_SUCCESS);
#endif
    (void)errcode;
  }
  void SaveCurrentContext() {
#if TENSORFLOW_USE_ROCM
    auto errcode = hipCtxGetCurrent(&context);
    assert(errcode == hipSuccess);
#else
    auto errcode = cuCtxGetCurrent(&context);
    assert(errcode == CUDA_SUCCESS);
#endif
    (void)errcode;
  }
};

se::Stream* GetOrCreateDefaultCudaStreamExecutorStream(int device_ordinal,
                                                       GpuStreamHandle stream) {
  static std::mutex m;
  static std::map<std::pair<int, GpuStreamHandle>, TableEntry> table;
  auto key = std::make_pair(device_ordinal, stream);
  std::lock_guard<std::mutex> l(m);
  auto it = table.find(key);
  if (it == table.end()) {
#if TENSORFLOW_USE_ROCM
    auto platform =
        se::MultiPlatformManager::PlatformWithName("ROCM").ValueOrDie();
    auto gpu_platform = static_cast<se::gpu::ROCmPlatform*>(platform);
#else
    auto platform =
        se::MultiPlatformManager::PlatformWithName("CUDA").ValueOrDie();
    auto gpu_platform = static_cast<se::gpu::CudaPlatform*>(platform);
#endif
    TableEntry e;
    bool use_multi_cuda_se = true;
    tensorflow::ReadBoolFromEnvVar("TAO_ENABLE_MULTIPLE_CUDA_STREAM_EXECUTOR",
                                   true, &use_multi_cuda_se);
    se::StreamExecutor* executor = nullptr;
    if (use_multi_cuda_se) {
      executor = gpu_platform->ExecutorForDevice(device_ordinal, (void*)stream)
                     .ValueOrDie();
    } else {
      executor = gpu_platform->ExecutorForDevice(device_ordinal).ValueOrDie();
    }
    auto cuda_executor = se::gpu::ExtractGpuExecutor(executor);
    // Not wrapper with a unique_ptr to avoid stream is destroyed after executor
    // StreamExecutor use cuda primary context.
    // A non-primary context was not verfied by StreamExectuor.
    // So we would not support non-primary cuda context currently.
    e.stream = new se::Stream(
        executor, std::make_unique<CUDAStream>(cuda_executor, stream), false);
    e.stream->Init();
    e.SaveCurrentContext();
    it = table.emplace(key, std::move(e)).first;
  }
  it->second.ActivateContext();
  return it->second.stream;
}

}  // namespace gpu
}  // namespace ral
}  // namespace tao

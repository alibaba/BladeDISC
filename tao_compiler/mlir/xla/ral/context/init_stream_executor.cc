#include "tensorflow/compiler/mlir/xla/ral/context/init_stream_executor.h"

#include "tensorflow/compiler/mlir/xla/ral/context/base/cuda/cuda_stream.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/stream_executor/stream.h"
#if TENSORFLOW_USE_ROCM
#include "tensorflow/stream_executor/rocm/rocm_platform.h"
#else
#include "tensorflow/stream_executor/cuda/cuda_platform.h"
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
    e.stream = new se::Stream(executor, new CUDAStream(cuda_executor, stream));
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

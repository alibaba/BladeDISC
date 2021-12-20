#pragma once
#include "tensorflow/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/stream_executor/gpu/gpu_types.h"

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

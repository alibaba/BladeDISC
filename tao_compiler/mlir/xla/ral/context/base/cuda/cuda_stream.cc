#include "tensorflow/compiler/mlir/xla/ral/context/base/cuda/cuda_stream.h"

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
    se::port::Status status = se::gpu::GpuDriver::DestroyEvent(
        parent_->gpu_context(), &completed_event_);
    if (!status.ok()) {
      LOG(ERROR) << status.error_message();
    }
  }
}

}  // namespace gpu
}  // namespace ral
}  // namespace tao

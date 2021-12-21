#ifndef RAL_CONTEXT_INIT_STREAM_EXECUTOR_H_
#define RAL_CONTEXT_INIT_STREAM_EXECUTOR_H_

#include "tensorflow/compiler/mlir/xla/ral/context/stream_executor_based_impl.h"
#include "tensorflow/stream_executor/gpu/gpu_types.h"
#include "tensorflow/stream_executor/stream.h"

namespace tao {
namespace ral {
namespace gpu {

using ::stream_executor::gpu::GpuStreamHandle;

::stream_executor::Stream* GetOrCreateDefaultCudaStreamExecutorStream(
    int device_ordinal = 0, GpuStreamHandle stream = nullptr);

}  // namespace gpu
}  // namespace ral
}  // namespace tao

#endif  // RAL_CONTEXT_INIT_STREAM_EXECUTOR_H_

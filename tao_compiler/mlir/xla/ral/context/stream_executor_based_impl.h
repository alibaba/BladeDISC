
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

//===- cuda_context_impl.h ----------------------===//
//
// Copyright 2020 The PAI Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================

#ifndef RAL_CONTEXT_BASE_CUDA_CUDA_CONTEXT_IMPL_H_
#define RAL_CONTEXT_BASE_CUDA_CUDA_CONTEXT_IMPL_H_

#if GOOGLE_CUDA
#include "cuda.h"
#endif

#if TENSORFLOW_USE_ROCM
#define __HIP_DISABLE_CPP_FUNCTIONS__
#include "rocm/include/hip/hip_runtime.h"
#endif

#include "mlir/xla/ral/context/base/cpu/cpu_context_impl.h"
#include "mlir/xla/ral/device/cpu/cpu_driver.h"
#include "mlir/xla/ral/device/gpu/gpu_driver.h"
#include "mlir/xla/ral/ral_context.h"

// Raw cuda ral implementation.

namespace tao {
namespace ral {
namespace gpu {

#if TENSORFLOW_USE_ROCM
using GpuStreamHandle = hipStream_t;
#else
using GpuStreamHandle = CUstream;
#endif

struct BaseCudaContextOption {
  GpuStreamHandle stream = nullptr;
  int device_ordinal = 0;
  bool use_stream_executor = true;
  bool cache_workspace_mem_across_execution = false;
  std::shared_ptr<Allocator> gpu_allocator;
};

std::unique_ptr<BaseContext> MakeBaseCudaContext(
    BaseContextOption& opt, ::tao::ral::cpu::BaseCpuContextOption& cpu_opt,
    ::tao::ral::gpu::BaseCudaContextOption& gpu_opt);

struct BaseCudaExecutionContext
    : public tao::ral::cpu::BaseCpuExecutionContext {
  BaseCudaExecutionContext(BaseContext* ctx);
  ~BaseCudaExecutionContext();

  // We need to sync on the gpu stream before we fetch the first output.
  bool synced = false;
  // all buffer allocated by the gpu_allocator
  std::unordered_map<const_buffer_t, int> device_ptr_map;

 protected:
  virtual void setOutputDeleter(OutputBufferWrapper& output) override;
};

}  // namespace gpu
}  // namespace ral
}  // namespace tao

#endif  // RAL_CONTEXT_BASE_CUDA_CUDA_CONTEXT_IMPL_H_

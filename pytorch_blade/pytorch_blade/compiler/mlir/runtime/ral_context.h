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

#include <memory>
#include <mutex>
#include <unordered_map>

#include <torch/script.h>

#ifdef TORCH_BLADE_BUILD_WITH_CUDA
#ifdef TORCH_BLADE_USE_ROCM
#define __HIP_PLATFORM_HCC__
#define TENSORFLOW_USE_ROCM 1
#include <c10/hip/HIPStream.h>
namespace c10 {
namespace cuda {
using CUDAStream = ::c10::hip::HIPStream;
} // namespace cuda
} // namespace c10
#else // TORCH_BLADE_USE_ROCM
#include <c10/cuda/CUDAStream.h>
#endif // TORCH_BLADE_USE_ROCM
#include "mlir/ral/context/base/cuda/cuda_context_impl.h"
#endif // TORCH_BLADE_BUILD_WITH_CUDA
// TODO(disc): figure out why the bazel does not trigger re-compile this file
// after we update ral.
#include "mlir/ral/context/base/cpu/cpu_context_impl.h"

#include "pytorch_blade/common_utils/macros.h"
#include "pytorch_blade/common_utils/tempfs.h"
#include "pytorch_blade/compiler/backends/engine_interface.h"
#include "pytorch_blade/compiler/jit/shape_type_spec.h"

namespace torch {
namespace blade {

class RalContext {
  using EntryFunc = std::function<void(void**)>;

 public:
  RalContext(std::shared_ptr<backends::EngineState> state);
  ~RalContext();

  at::List<at::Tensor> Execute(const at::List<at::Tensor>&);

 private:
  void BindingInputs(
      const at::List<at::Tensor>& inputs,
      tao::ral::ExecutionContext& exec_ctx);
  void CheckCurrentDevice(const at::List<at::Tensor>& inputs);
  at::List<at::Tensor> CreateAndBindingOutputs(
      tao::ral::ExecutionContext& exec_ctx);
  at::List<at::Tensor> PreProcessInputs(const at::List<at::Tensor>& inputs);
  std::tuple<void*, void*> LoadEngine(const std::string& ral_engine_bytes);

  std::shared_ptr<backends::EngineState> engine_state_;
  TempFile lib_tmpf_{"ral_lib.so"};
  TempFile meta_tmpf_{"ral_meta.pb"};

#ifdef TORCH_BLADE_BUILD_WITH_CUDA
  int64_t LazyInitCurrentDevice();

  constexpr static int64_t NULL_GPU_DEVICE = -1;
  std::atomic<int64_t> gpu_device_{NULL_GPU_DEVICE};
  std::mutex mtx_;
  std::unordered_map<
      c10::cuda::CUDAStream,
      std::unique_ptr<tao::ral::BaseContext>>
      ral_ctx_map_;
  tao::ral::BaseContext* LoadCache();
#else
  std::unique_ptr<tao::ral::BaseContext> ral_ctx_;
#endif // TORCH_BLADE_BUILD_WITH_CUDA

  tao::ral::BaseContextOption default_opt_;
  tao::ral::cpu::BaseCpuContextOption cpu_opt_;
  void* tao_lib_;
  EntryFunc entry_func_;
};

} // namespace blade
} // namespace torch

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
#include "tensorflow/compiler/mlir/xla/ral/context/base/cuda/cuda_context_impl.h"
#endif // TORCH_BLADE_BUILD_WITH_CUDA
#include "tensorflow/compiler/mlir/xla/ral/context/base/cpu/cpu_context_impl.h"

#include "common_utils/macros.h"
#include "common_utils/tempfs.h"
#include "compiler/jit/shape_type_spec.h"

namespace torch {
namespace blade {

class RalContext {
  using EntryFunc = std::function<void(void**)>;

 public:
  RalContext(
      const std::string& ral_engine_bytes,
      const std::string& ral_const_bytes,
      const std::string& input_type_spec,
      const std::string& output_type_spec,
      const std::string& input_dev_str,
      const std::string& output_dev_str);
  ~RalContext();

  torch::List<torch::Tensor> Forward(const torch::List<torch::Tensor>&);

 private:
  void BindingInputs(
      const torch::List<torch::Tensor>& inputs,
      tao::ral::ExecutionContext& exec_ctx) const;
  bool CheckCurrentDevice(const torch::List<torch::Tensor>& inputs) const;
  torch::List<torch::Tensor> CreateAndBindingOutputs(
      tao::ral::ExecutionContext& exec_ctx) const;
  torch::List<torch::Tensor> PreProcessInputs(
      const torch::List<torch::Tensor>& inputs) const;
  std::tuple<void*, void*> LoadEngine(const std::string& ral_engine_bytes);
  ShapeTypeSpec input_type_spec_;
  ShapeTypeSpec output_type_spec_;
  std::vector<std::string> input_dev_;
  std::vector<std::string> output_dev_;

  TempFile lib_tmpf_;
  TempFile meta_tmpf_;

#ifdef TORCH_BLADE_BUILD_WITH_CUDA
  int64_t gpu_device_;
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

// Copyright 2022 The BladeDISC Authors. All rights reserved.
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

#include <ATen/core/List.h>
#include <ATen/core/ScalarType.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>

#include <atomic>
#include "pytorch_blade/common_utils/macros.h"
#include "pytorch_blade/compiler/backends/engine_interface.h"
#include "pytorch_blade/compiler/tensorrt/bridge/tensorrt_common.h"

namespace torch {
namespace blade {
namespace tensorrt {
at::ScalarType NvDataType2TorchDataType(nvinfer1::DataType dtype);

class TRTContext {
  using State = torch::blade::backends::EngineState;
  using SingleProfileBindingIndex = std::vector<int>;

 public:
  DISALLOW_COPY_AND_ASSIGN(TRTContext);
  // this is not thread-safe initialization
  TRTContext(std::shared_ptr<State> state);
  std::string SerializeAsString() const;
  at::List<at::Tensor> Execute(const at::List<at::Tensor>&);
  bool IsInRange(const at::List<at::Tensor>& inputs);

 private:
  // Setup binding buffers to the input/output blobs on the GPU.
  // Input/output blob mem ptr should be provided by the caller,
  // the TRTEngine is not the owner of the blobs' cuda memory.
  void BindingInputs(const at::List<at::Tensor>& inputs, std::vector<void*>&)
      const;
  at::List<at::Tensor> CreateAndBindingOutputs(
      std::vector<void*>&,
      std::shared_ptr<nvinfer1::IExecutionContext>& context) const;
  at::List<at::Tensor> PostProcessOutputs(
      const at::List<at::Tensor>& outputs) const;
  at::List<at::Tensor> PreProcessInputs(
      const at::List<at::Tensor>& inputs,
      std::shared_ptr<nvinfer1::IExecutionContext>& context);
  bool ChangingShape(
      const at::List<at::Tensor>& inputs,
      std::shared_ptr<nvinfer1::IExecutionContext>& context);

  bool CheckCurrentDevice(const at::List<at::Tensor>& inputs) const;
  std::shared_ptr<nvinfer1::IExecutionContext> GetExecutionContext(
      c10::cuda::CUDAStream& stream,
      const at::List<at::Tensor>& inputs);
  void UpdateProfileIfNeed(const at::List<at::Tensor>& inputs);
  bool IsInRange(const at::List<at::Tensor>&, int64_t);
  int64_t tensorrt_device_;
  mutable std::mutex lock_;
  TrtUniquePtr<nvinfer1::ICudaEngine> engine_;
  std::unordered_map<
      c10::cuda::CUDAStream,
      std::vector<std::shared_ptr<nvinfer1::IExecutionContext>>>
      contexts_map_;
  std::vector<SingleProfileBindingIndex> input_bind_indices_;
  std::vector<SingleProfileBindingIndex> output_bind_indices_;
  std::shared_ptr<State> engine_state_;

  int optimization_profile_ = 0;
  int profile_num_ = 1;
};
} // namespace tensorrt
} // namespace blade
} // namespace torch

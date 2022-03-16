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

#include <c10/cuda/CUDAStream.h>
#include <atomic>
#include "common_utils/macros.h"
#include "compiler/backends/engine_interface.h"
#include "compiler/tensorrt/bridge/tensorrt_common.h"

namespace torch {
namespace blade {
namespace tensorrt {
torch::ScalarType NvDataType2TorchDataType(nvinfer1::DataType dtype);

class TRTContext {
  using State = torch::blade::backends::EngineState;
  using SingleProfileBindingIndex = std::vector<int>;

 public:
  DISALLOW_COPY_AND_ASSIGN(TRTContext);
  // this is not thread-safe initialization
  TRTContext(std::shared_ptr<State> state);
  std::string SerializeAsString() const;
  torch::List<torch::Tensor> Execute(const torch::List<torch::Tensor>&);

 private:
  // Setup binding buffers to the input/output blobs on the GPU.
  // Input/output blob mem ptr should be provided by the caller,
  // the TRTEngine is not the owner of the blobs' cuda memory.
  void BindingInputs(
      const torch::List<torch::Tensor>& inputs,
      std::vector<void*>&) const;
  torch::List<torch::Tensor> CreateAndBindingOutputs(
      std::vector<void*>&,
      std::shared_ptr<nvinfer1::IExecutionContext>& context) const;
  torch::List<torch::Tensor> PostProcessOutputs(
      const torch::List<torch::Tensor>& outputs) const;
  torch::List<torch::Tensor> PreProcessInputs(
      const torch::List<torch::Tensor>& inputs,
      std::shared_ptr<nvinfer1::IExecutionContext>& context);
  bool ChangingShape(
      const torch::List<torch::Tensor>& inputs,
      std::shared_ptr<nvinfer1::IExecutionContext>& context);

  bool CheckCurrentDevice(const torch::List<torch::Tensor>& inputs) const;
  std::shared_ptr<nvinfer1::IExecutionContext> GetExecutionContext(
      c10::cuda::CUDAStream& stream,
      const torch::List<torch::Tensor>& inputs);
  void UpdateProfileIfNeed(const torch::List<torch::Tensor>& inputs);
  bool IsInRange(const torch::List<torch::Tensor>&, int64_t);

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

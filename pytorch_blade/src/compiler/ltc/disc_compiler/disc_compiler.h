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

#include <ATen/Functions.h>
#include <sys/stat.h>
#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/ts_backend/ts_lowering_context.h>
#include <torch/script.h>
#include <unistd.h>

namespace torch_disc {
namespace compiler {

// Executable represents an executable program
class Executable {
 public:
  Executable(
      const std::shared_ptr<torch::jit::Graph>& graph,
      const std::vector<c10::IValue>& disc_inputs)
      : graph_(graph), graph_executor_(graph, ""), disc_inputs_(disc_inputs) {}

  std::vector<torch::lazy::BackendDataPtr> Run(
      c10::ArrayRef<torch::lazy::BackendDataPtr> arguments,
      const torch::lazy::BackendDevice& device,
      bool default_device_is_cuda);

  std::shared_ptr<torch::jit::Graph> graph() {
    return graph_;
  }

 private:
  std::shared_ptr<torch::jit::Graph> graph_;
  torch::jit::GraphExecutor graph_executor_;
  std::vector<c10::IValue> disc_inputs_;
};

using ExecutablePtr = std::shared_ptr<Executable>;

ExecutablePtr CompileToDiscExecutable(
    const std::shared_ptr<torch::jit::Graph>& graph,
    c10::ArrayRef<torch::lazy::BackendDataPtr> arguments);

} //  namespace compiler
} //  namespace torch_disc

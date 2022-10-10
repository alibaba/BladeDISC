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

#include <ATen/core/ivalue.h>
#include <sys/stat.h>
#include <unistd.h>

namespace torch {
namespace jit {
class Graph;
class GraphExecutor;
} // namespace jit
namespace lazy {
class BackendData;
class BackendDevice;
} // namespace lazy
} // namespace torch

namespace torch_disc {
namespace compiler {

// Executable represents an executable program
class Executable {
 public:
  Executable(
      const std::shared_ptr<torch::jit::Graph>& graph,
      const std::vector<c10::IValue>& disc_inputs);

  std::vector<std::shared_ptr<torch::lazy::BackendData>> Run(
      c10::ArrayRef<std::shared_ptr<torch::lazy::BackendData>> arguments,
      const torch::lazy::BackendDevice& device,
      bool default_device_is_cuda);

  std::shared_ptr<torch::jit::Graph> graph() {
    return graph_;
  }

 private:
  std::shared_ptr<torch::jit::Graph> graph_;
  std::shared_ptr<torch::jit::GraphExecutor> graph_executor_;
  std::vector<c10::IValue> disc_inputs_;
  bool is_first_run_ = true;
};

using ExecutablePtr = std::shared_ptr<Executable>;

ExecutablePtr CompileToDiscExecutable(
    const std::shared_ptr<torch::jit::Graph>& graph,
    c10::ArrayRef<std::shared_ptr<torch::lazy::BackendData>> arguments);

} //  namespace compiler
} //  namespace torch_disc

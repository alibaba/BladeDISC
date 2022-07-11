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

#include "ltc/disc_compiler/disc_compiler.h"

#include <ATen/Functions.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/ts_backend/ts_backend_impl.h>
#include <torch/csrc/lazy/ts_backend/ts_lowering_context.h>
#include <torch/script.h>
#include "common_utils/utils.h"
#include "compiler/jit/torch/shape_analysis.h"
#include "ltc/disc_compiler/passes/disc_fuser.h"
#include "ltc/disc_compiler/passes/register_disc_class.h"

namespace torch_disc {
namespace compiler {
using TSData = torch::lazy::TSData;

Executable::Executable(
    const std::shared_ptr<torch::jit::Graph>& graph,
    const std::vector<c10::IValue>& disc_inputs)
    : graph_(graph), disc_inputs_(disc_inputs) {
  graph_executor_ = std::make_shared<torch::jit::GraphExecutor>(graph, "");
}

std::vector<torch::lazy::BackendDataPtr> Executable::Run(
    c10::ArrayRef<torch::lazy::BackendDataPtr> arguments,
    const torch::lazy::BackendDevice& device,
    bool default_device_is_cuda) {
  std::vector<c10::IValue> stack;
  for (auto argument : arguments) {
    const auto ts_data = std::static_pointer_cast<TSData>(argument);
    if (ts_data->scalar.has_value()) {
      stack.emplace_back(ts_data->scalar.value());
    } else {
      // TODO(whc) should this check be made more general? it's written somewhat
      // oddly
      CHECK(
          !default_device_is_cuda ||
          ts_data->data().device().type() == at::kCUDA);
      stack.emplace_back(ts_data->data());
    }
  }
  stack.insert(stack.end(), disc_inputs_.begin(), disc_inputs_.end());
  graph_executor_->run(stack);

  std::vector<torch::lazy::BackendDataPtr> results;
  for (torch::jit::IValue component : stack) {
    at::Tensor result = component.toTensor();
    at::IntArrayRef result_sizes = result.sizes();
    torch::lazy::Shape shape(
        result.scalar_type(),
        std::vector<int64_t>(result_sizes.begin(), result_sizes.end()));
    results.push_back(std::make_shared<TSData>(result, shape, device));
  }
  return results;
}

void EnhancementInputShape(
    const std::shared_ptr<torch::jit::Graph>& graph,
    c10::ArrayRef<torch::lazy::BackendDataPtr> arguments) {
  for (size_t i = 0; i < arguments.size(); ++i) {
    auto argument = arguments[i];
    auto input = graph->inputs()[i];
    const auto ts_data = std::static_pointer_cast<TSData>(argument);

    if (ts_data->HasValue()) {
      input->setType(c10::TensorType::create(ts_data->data()));
    }
  }
}

ExecutablePtr CompileToDiscExecutable(
    const std::shared_ptr<torch::jit::Graph>& graph,
    c10::ArrayRef<torch::lazy::BackendDataPtr> arguments) {
  bool disable_disc = torch::blade::env::ReadBoolFromEnvVar(
      "TORCH_DISC_LTC_DISABLE_DISC", false);
  if (disable_disc) {
    auto disc_inputs = std::vector<c10::IValue>{};
    return std::make_shared<Executable>(graph, disc_inputs);
  }
  EnhancementInputShape(graph, arguments);
  // Inference shape
  torch::blade::PropagateInputShapes(graph);
  // cluster disc compitable nodes into a sub-graph
  GRAPH_DEBUG("before DiscFusion\n ", *graph);
  DiscFusion(graph);
  GRAPH_DEBUG("after DiscFusion, before EliminateDeadCode\n", *graph)
  torch::jit::EliminateDeadCode(graph);
  // register a disc custom class to run RAL at runtime stage
  GRAPH_DEBUG("after EliminateDeadCode, before RegisterDiscClass\n", *graph);
  auto disc_inputs = RegisterDiscClass(graph);
  GRAPH_DEBUG("after RegisterDiscClass\n", *graph);
  GRAPH_DUMP("after CompileToDiscExecutable\n", graph);
  return std::make_shared<Executable>(graph, disc_inputs);
}

} //  namespace compiler
} //  namespace torch_disc

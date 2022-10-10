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

#include "pytorch_blade/ltc/disc_compiler/disc_compiler.h"

#include <ATen/Functions.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/core/hash.h>
#include <torch/csrc/lazy/ts_backend/ts_backend_impl.h>
#include <torch/csrc/lazy/ts_backend/ts_lowering_context.h>
#include <torch/script.h>
#include "pytorch_blade/common_utils/utils.h"
#include "pytorch_blade/compiler/jit/torch/shape_analysis.h"
#include "pytorch_blade/ltc/disc_compiler/passes/disc_fuser.h"
#include "pytorch_blade/ltc/disc_compiler/passes/register_disc_class.h"
#include "pytorch_blade/ltc/disc_compiler/replay.h"

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
  if (is_first_run_ && IsEnableReplayToolkit()) {
    BeginClusterReplayRecord();
  }
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
  if (is_first_run_) {
    EndClusterReplayRecord();
    is_first_run_ = false;
  }
  return results;
}

void EnhancementInputShape(
    const std::shared_ptr<torch::jit::Graph>& graph,
    c10::ArrayRef<torch::lazy::BackendDataPtr> arguments) {
  for (size_t i = 0; i < arguments.size(); ++i) {
    auto argument = arguments[i];
    auto input = graph->inputs()[i];
    const auto ts_data =
        std::static_pointer_cast<torch::lazy::TSData>(argument);
    if (ts_data->HasValue()) {
      if (torch::blade::env::ReadBoolFromEnvVar(
              "TORCH_DISC_DYNAMIC_SHAPE_COMPILE", false)) {
        auto t = ts_data->data();
        size_t rank = t.sizes().size();
        input->setType(c10::TensorType::create(
            t.scalar_type(),
            t.device(),
            c10::SymbolicShape(c10::optional<size_t>(rank)),
            c10::VaryingShape<c10::Stride>(rank),
            t.requires_grad()));
      } else {
        input->setType(c10::TensorType::create(ts_data->data()));
      }
    }
  }
}

void RecordReplay(
    const std::shared_ptr<torch::jit::Graph>& graph,
    c10::ArrayRef<torch::lazy::BackendDataPtr> arguments) {
  auto disc_hash = torch::lazy::DataHash(graph.get(), sizeof(*graph.get()));
  auto disc_hash_str = torch::lazy::HashToString(disc_hash);
  std::string dump_path = "/tmp/replay_lazy_ts_" + disc_hash_str;
  TORCH_CHECK(
      !mkdir(dump_path.c_str(), 0755), "unable to create dir: " + dump_path);
  VLOG(0) << "replay toolkit dump TorchScript program and data on: "
          << dump_path;
  ::torch_disc::compiler::DumpProgramAndData(
      graph->copy(), arguments, dump_path);
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
  // record program and data for replay toolkit
  if (IsEnableReplayToolkit())
    RecordReplay(graph, arguments);

  GRAPH_DEBUG("before PropagateInputShapes\n", *graph);
  EnhancementInputShape(graph, arguments);
  // Inference shape
  torch::blade::PropagateInputShapes(graph);
  // cluster disc compitable nodes into a sub-graph
  GRAPH_DEBUG("after PropagateInputShapes, before DiscFusion\n ", *graph);
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

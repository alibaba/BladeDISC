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

#include "pytorch_blade/ltc/disc_compiler/replay.h"

#include <sys/stat.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/lazy/ts_backend/ts_backend_impl.h>
#include <torch/script.h>
#include <fstream>

#ifdef TORCH_BLADE_BUILD_WITH_CUDA
#include <cuda_profiler_api.h>
#endif

#include "pytorch_blade/common_utils/logging.h"
#include "pytorch_blade/common_utils/utils.h"
#include "pytorch_blade/compiler/jit/tool_funcs.h"
#include "pytorch_blade/ltc/disc_compiler/disc_compiler.h"

namespace torch_disc {
namespace compiler {

void FoldOutputs(std::shared_ptr<torch::jit::Graph> graph) {
  auto return_node = graph->return_node();
  auto tuple = return_node->input()->node();
  return_node->removeAllInputs();
  for (auto input : tuple->inputs()) {
    return_node->addInput(input);
  }
}

void FusionOutputs(std::shared_ptr<torch::jit::Graph> graph) {
  auto return_node = graph->return_node();

  auto list_construct = graph->insertNode(graph->createList(
      torch::jit::OptionalType::ofTensor(), return_node->inputs()));
  auto tensor_type = at::TensorType::get();
  auto list_type = at::ListType::create(tensor_type);
  list_construct->output()->setType(list_type);
  list_construct->moveBefore(return_node);
  return_node->removeAllInputs();
  return_node->addInput(list_construct->output());
}

std::shared_ptr<torch::jit::Graph> loadGraph(torch::jit::Module& module) {
  const auto method_name =
      torch::QualifiedName(*module.type()->name(), "forward");
  auto func = module._ivalue()->compilation_unit()->find_function(method_name);
  auto graph = torch::jit::tryToGraphFunction(*func)->graph();
  return graph;
}

std::vector<torch::lazy::BackendDataPtr> loadArguments(
    const std::string& path) {
  int k = 0;
  std::string fname = path + "/" + std::to_string(k) + ".pt";
  struct stat finfo;
  std::vector<torch::lazy::BackendDataPtr> stack;
  while (true) {
    if (stat(fname.c_str(), &finfo) != 0)
      break;
    std::ifstream input_stream(fname, std::ios::in | std::ios::binary);
    std::vector<char> input(
        (std::istreambuf_iterator<char>(input_stream)),
        std::istreambuf_iterator<char>());
    auto ivalue = torch::jit::pickle_load(input);
    auto cuda_device =
        torch::lazy::getBackend()->GetBackendDevice(c10::Device(c10::kCUDA, 0));
    if (ivalue.isScalar()) {
      auto data = torch::lazy::getBackend()->MakeComputationDataFromScalar(
          ivalue.toScalar(), cuda_device);
      stack.emplace_back(data);
    } else if (ivalue.isTensor()) {
      auto tensor = ivalue.toTensor();
      auto shape = torch::lazy::Shape(tensor.scalar_type(), tensor.sizes());
      auto data = torch::lazy::getBackend()->MakeComputationDataFromTensor(
          tensor, shape, cuda_device);
      stack.emplace_back(data);
    } else {
      TORCH_CHECK(false, "arguments only support [scalar, tensor] type.");
    }
    fname = path + "/" + std::to_string(++k) + ".pt";
  }
  std::cout << "finish loading TorchScript graph and data:" << stack.size()
            << std::endl;
  return stack;
}

void LoadAndReplay(
    const std::string& path,
    int iters,
    int warmup,
    bool whole_graph) {
  std::string module_path = path + "/graph.pt";
  auto module = torch::jit::load(module_path);
  auto arguments = loadArguments(path);
  auto graph = loadGraph(module);
  graph->eraseInput(0);
  torch::jit::ConstantPropagation(graph);
  FoldOutputs(graph);
  torch::jit::EliminateDeadCode(graph);

  auto executable = CompileToDiscExecutable(graph, arguments);
  auto cuda_device =
      torch::lazy::getBackend()->GetBackendDevice(c10::Device(c10::kCUDA, 0));
  for (int i = 0; i < warmup; ++i)
    executable->Run(arguments, cuda_device, /*default device is cuda*/ true);
  {
    std::stringstream ss;
    ss << "warmup: " << warmup << " iters: " << iters << ", average cost: ";
    Timer time(ss.str(), iters);
    for (int i = 0; i < iters; ++i) {
      {
        std::stringstream ss2;
        ss2 << "the [" << i << "] iteration cost: ";
        Timer t2(ss2.str(), 1);
        executable->Run(
            arguments, cuda_device, /*default device is cuda*/ true);
      }
    }
  }
#ifdef TORCH_BLADE_BUILD_WITH_CUDA
  if (torch::blade::env::ReadBoolFromEnvVar(
          "TORCH_DISC_REPLAY_ENABLE_NVPROF", true)) {
    cudaProfilerStart();
    std::stringstream ss;
    ss << "spent with profiler: ";
    Timer time(ss.str(), 1);
    executable->Run(arguments, cuda_device, /*default device is cuda*/ true);
    cudaProfilerStop();
  }
#endif
}

void DumpProgramAndData(
    const std::shared_ptr<torch::jit::Graph> orig_graph,
    c10::ArrayRef<std::shared_ptr<torch::lazy::BackendData>> arguments,
    const std::string& path) {
  auto graph = orig_graph->copy();

  auto module_path = path + "/graph.pt";
  torch::jit::Module module("__torch__.PlaceholderModule");
  module.register_attribute("training", torch::jit::BoolType::get(), true);
  // module.save requires one output
  FusionOutputs(graph);
  torch::blade::create_method_from_graph(module, "forward", graph);
  module.save(module_path);

  std::vector<c10::IValue> stack;
  for (auto argument : arguments) {
    const auto ts_data =
        std::static_pointer_cast<torch::lazy::TSData>(argument);
    if (ts_data->scalar.has_value()) {
      stack.emplace_back(ts_data->scalar.value());
    } else {
      stack.emplace_back(ts_data->data());
    }
  }
  torch::blade::DumpIValues(stack, path);
}

torch::jit::Module ConvertGraphToModule(
    const std::shared_ptr<torch::jit::Graph>& graph) {
  torch::jit::Module module("__torch__.PlaceholderModule");
  module.register_attribute("training", torch::jit::BoolType::get(), true);
  FusionOutputs(graph);
  for (auto& value : graph->inputs()) {
    value->setDebugName("arg_" + value->debugName());
  }
  torch::blade::create_method_from_graph(module, "forward", graph);
  return module;
}

bool IsEnableReplayToolkit() {
  return torch::blade::env::ReadBoolFromEnvVar(
      "TORCH_DISC_ENABLE_REPLAY", false);
}

bool IsEnableClusterReplayRecord() {
  return torch::blade::env::ReadBoolFromEnvVar(
      "TORCH_DISC_ENABLE_REPLAY_ON_CLUSTER", false);
}

void BeginClusterReplayRecord() {
  setenv("TORCH_DISC_ENABLE_REPLAY_ON_CLUSTER", "true", true);
}

void EndClusterReplayRecord() {
  unsetenv("TORCH_DISC_ENABLE_REPLAY_ON_CLUSTER");
}

bool IsForceFallback() {
  return torch::blade::env::ReadBoolFromEnvVar(
      "TORCH_DISC_FORCE_FALLBACK", false);
}

} //  namespace compiler
} //  namespace torch_disc

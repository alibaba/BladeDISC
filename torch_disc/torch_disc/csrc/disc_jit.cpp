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

#include "torch_disc/csrc/disc_jit.h"

#include <ATen/Functions.h>
#include <sys/stat.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/ts_backend/ts_lowering_context.h>
#include <torch/script.h>
#include <unistd.h>

#include <sstream>

#include "common_utils/tempfs.h"
#include "compiler/jit/fusion.h"
#include "compiler/mlir/converters/mhlo_conversion.h"
#include "lazy_tensor_core/csrc/ts_backend/backend_impl.h"
#include "torch_disc/csrc/backend_impl.h"

namespace torch_disc {
namespace compiler {

using namespace ::torch::jit;

std::vector<Node*> FakeCluster(const std::shared_ptr<Graph>& graph) {
  // TODO(yancey.yx):
  // This a very rough implementation to go thought the whole DiscBackend
  // pipeline, this function only cluster one node that is supported on mhlo
  // conversion module into a group. We should re-implement this.
  std::vector<Node*> nodes;
  for (auto node : graph->nodes()) {
    if (torch::blade::IsMlirMhloSupported(*node) &&
        node->kind() != prim::Constant) {
      nodes.push_back(node);
    }
  }
  return nodes;
}

void CastToTensorInputs(const std::shared_ptr<Graph>& graph, Node* sub_graph,
                        Graph* disc_node) {
  for (size_t i = 0; i < sub_graph->inputs().size(); ++i) {
    auto input = sub_graph->inputs()[i];
    if (input->type()->cast<c10::TensorType>() == nullptr) {
      auto cast_tensor = graph->createNumToTensor(input);
      cast_tensor->insertAfter(input->node());
      sub_graph->replaceInput(i, cast_tensor->output());
      // disc_node->replaceInput(i, cast_tensor->output());
    }
  }
  std::cout << "After [CastToTensorInputs]: \n"
            << graph->toString() << std::endl;
}

void FusionDiscNodes(const std::shared_ptr<Graph>& graph) {
  auto nodes = FakeCluster(graph);
  for (auto node : nodes) {
    // create a sub-graph
    auto group = graph->createWithSubgraph(prim::FusionGroup);
    auto sub_graph = group->g(attr::Subgraph);
    group->insertAfter(node);
    auto node_merged = torch::blade::MergeNodeIntoGroup(group, node);

    auto outputs = node->outputs();
    auto new_outputs = node_merged->outputs();

    for (int i = 0; i < outputs.size(); ++i) {
      auto out = outputs[i];
      sub_graph->registerOutput(new_outputs[i]);
      auto group_out = group->addOutput();
      out->replaceAllUsesWith(group_out);
      group_out->setType(out->type());
    }
    node->destroy();
  }
  torch::jit::EliminateDeadCode(graph);
}

std::string DiscCMD(const std::string& mlir_fname,
                    const std::string& out_fname) {
  std::stringstream ss;
  std::string logf = mlir_fname + ".log";
  ss << "./disc_compiler_main " << mlir_fname << " " << out_fname << " > "
     << logf << " 2>&1 ";
  return ss.str();
}

std::string GetTempDirectory(std::string dir) {
  auto tid = std::this_thread::get_id();
  uint64_t pid = getpid();
  uint64_t us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now().time_since_epoch())
          .count();
  std::stringstream ss;
  ss << dir << "/" << tid << "-" << pid << "-" << us;
  TORCH_CHECK(!mkdir(ss.str().c_str(), 0755),
              "unable to create dir: " + ss.str());
  return ss.str();
}

std::string MhloConversaion(const std::shared_ptr<Graph>& graph) {
  std::string parsable_mlir;
  std::string pretty_mlir;
  std::string input_dev_str;
  std::string output_dev_str;
  std::tie(parsable_mlir, pretty_mlir, input_dev_str, output_dev_str) =
      torch::blade::ConvertTorchScriptToMhlo(graph);
  std::string dir = GetTempDirectory("/tmp");
  std::string in_fname = dir + "/disc.mlir";
  std::ofstream outfile(in_fname);
  outfile << parsable_mlir << std::endl;
  outfile.flush();
  std::cout << "=========" << std::endl;
  std::cout << graph->toString() << std::endl;
  std::cout << parsable_mlir << std::endl
            << "====================" << std::endl;
  return in_fname;
}

void CallDiscCompiler(const std::string& mlir_fname) {
  std::string out_fname = mlir_fname + ".out";
  std::string cmd = DiscCMD(mlir_fname, out_fname);
  TORCH_CHECK(std::system(cmd.c_str()) == 0,
              "disc compile failed with cmd: " + cmd);
}

void DiscCompilation(const std::shared_ptr<Graph>& graph) {
  for (auto node : graph->nodes()) {
    if (node->kind() == prim::FusionGroup) {
      auto sub_graph = node->g(attr::Subgraph);
      std::string fname = MhloConversaion(sub_graph);
      CallDiscCompiler(fname);
    }
  }
}

void InferShapes(const std::shared_ptr<Graph>& graph,
                 c10::ArrayRef<torch::lazy::BackendDataPtr> arguments) {
  for (size_t i = 0; i < graph->inputs().size(); ++i) {
    auto argument = arguments[i];
    auto input = graph->inputs()[i];
    const auto ts_data = std::static_pointer_cast<TSData>(argument);

    if (ts_data->HasValue()) {
      std::cout << "set input as tensor type: " << input->debugName()
                << std::endl;
      input->setType(c10::TensorType::create(ts_data->data()));
      std::cout << int(input->type()->cast<c10::TensorType>() != nullptr)
                << std::endl;
    }
  }
  torch::jit::PropagateInputShapes(graph);
}

void DiscJIT(torch::lazy::TSComputation& computation,
             c10::ArrayRef<torch::lazy::BackendDataPtr> arguments) {
  auto graph = computation.graph()->copy();
  // 1. clustering and group DISC nodes
  // 2. conversion from torchscript Graph to mhlo Dialect on DISC nodes
  // 2. register DISC engine
  InferShapes(graph, arguments);
  FusionDiscNodes(graph);
  std::cout << "After FakeCluster: \n" << graph->toString() << std::endl;
  DiscCompilation(graph);
  std::cout << "After MhloConversion: \n" << graph->toString() << std::endl;
}

}  //  namespace compiler
}  //  namespace torch_disc
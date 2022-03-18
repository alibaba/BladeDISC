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
using TSData = torch_lazy_tensors::compiler::TSData;

// TODO(Yancey1989):
// This a very rough implementation to go thought the whole DiscBackend,
// this fake function only cluster one node that is supported on mhlo
// conversion module into a group. We should re-implement this.
std::vector<Node*> FakeCluster(const std::shared_ptr<Graph>& graph) {
  std::vector<Node*> nodes;
  for (auto node : graph->nodes()) {
    if (torch::blade::IsMlirMhloSupported(*node) &&
        node->kind() != prim::Constant) {
      nodes.push_back(node);
    }
  }
  return nodes;
}

// Give:
//  with prim::FusionGroup(
//      %1: Scalar,
//      %2: Scalar):
//    %3 Tensor = aten::add(%1, %2)
//  return %3
//
// Execute: CastToTensorInputs(sub_graph)
//
// After:
//  with prim::FusionGroup(
//      %1.1: Tensor,
//      %2.1: Tensor):
//    %4 = aten::item(%1.1, 1)
//    %5 = aten::item(%2.1, 1)
//    %3 Tensor = aten::add(%4, %5)
//    return %3
void DiscSubgraphInputsCast(Graph* disc_graph, size_t i) {
  auto new_input = disc_graph->insertInput(
      i, c10::string(disc_graph->inputs()[i]->debugName() + ".1"));
  auto orig_input = disc_graph->inputs()[i + 1];
  auto item_node = disc_graph->create(aten::item, {new_input});
  // TODO(Yancey1989): supports more types
  item_node->output()->setType(c10::IntType::get());
  disc_graph->appendNode(item_node);
  orig_input->replaceAllUsesWith(item_node->output());
  item_node->moveBefore(item_node->output()->uses()[0].user);
  disc_graph->eraseInput(i + 1);
}

void CastToTensorInputs(const std::shared_ptr<Graph>& graph, Node* sub_graph,
                        Node* disc_node) {
  auto disc_graph = disc_node->owningGraph();

  size_t inputs = sub_graph->inputs().size();
  for (size_t i = 0; i < inputs; ++i) {
    auto input = sub_graph->inputs()[i];
    if (input->type()->cast<c10::TensorType>() == nullptr) {
      auto cast_tensor = graph->createNumToTensor(input);
      cast_tensor->insertAfter(input->node());
      sub_graph->replaceInput(i, cast_tensor->output());

      // TODO(Yancey1989): cast output
      DiscSubgraphInputsCast(disc_graph, i);
    }
  }
  LOG(WARNING) << "After [CastToTensorInputs]: \n"
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
    CastToTensorInputs(graph, group, node_merged);
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
  TORCH_CHECK(!parsable_mlir.empty(), "mhlo conversaion failed!");

  std::string dir = GetTempDirectory("/tmp");
  std::string in_fname = dir + "/disc.mlir";
  std::ofstream outfile(in_fname);
  outfile << parsable_mlir << std::endl;
  outfile.flush();
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
      input->setType(c10::TensorType::create(ts_data->data()));
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
  LOG(WARNING) << "After FusionDiscNodes: \n" << graph->toString() << std::endl;
  DiscCompilation(graph);
  LOG(WARNING) << "After MhloConversion: \n" << graph->toString() << std::endl;
}

}  //  namespace compiler
}  //  namespace torch_disc
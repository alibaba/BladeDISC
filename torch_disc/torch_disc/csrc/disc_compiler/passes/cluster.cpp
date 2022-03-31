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

#include "torch_disc/csrc/disc_compiler/passes/cluster.h"

#include "compiler/jit/fusion.h"
#include "compiler/mlir/converters/mhlo_conversion.h"

namespace torch_disc {
namespace compiler {
using namespace ::torch::jit;

// TODO(Yancey1989):
// This a very rough implementation to go thought the whole DiscBackend,
// this fake function only cluster one node that is supported on mhlo
// conversion module into a group. We should re-implement this.
std::vector<Node*> FakeCluster(const std::shared_ptr<Graph>& graph) {
  std::vector<Node*> nodes;
  int cnt = 0;
  for (auto node : graph->nodes()) {
    if (torch::blade::IsMlirMhloSupported(*node) &&
        node->kind() != prim::Constant) {
      if (node->kind() == aten::addmm) continue;
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
void CastBoundaryScalarToTensor(Graph* disc_graph, size_t i) {
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

void CastGraphInputsToTensor(const std::shared_ptr<Graph>& graph,
                             Node* sub_graph, Node* disc_node) {
  auto disc_graph = disc_node->owningGraph();

  size_t inputs = sub_graph->inputs().size();
  for (size_t i = 0; i < inputs; ++i) {
    auto input = sub_graph->inputs()[i];
    if (input->type()->cast<c10::TensorType>() == nullptr) {
      auto cast_tensor = graph->createNumToTensor(input);
      cast_tensor->insertAfter(input->node());
      sub_graph->replaceInput(i, cast_tensor->output());

      // TODO(Yancey1989): cast output
      CastBoundaryScalarToTensor(disc_graph, i);
    }
  }
}

void ClusterDiscNodes(const std::shared_ptr<Graph>& graph) {
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
    CastGraphInputsToTensor(graph, group, node_merged);
  }
}

}  //  namespace compiler
}  //  namespace torch_disc
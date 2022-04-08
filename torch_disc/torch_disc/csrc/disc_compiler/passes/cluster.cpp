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

  std::copy_if(graph->nodes().begin(), graph->nodes().end(),
               std::back_inserter(nodes), [](torch::jit::Node* node) {
                 return torch::blade::IsMlirMhloSupported(*node) &&
                        node->kind() != prim::Constant;
               });

  return nodes;
}

// Give:
//  with prim::FusionGroup(
//      %p0: Tensor,
//      %p1: Tensor,
//      %p2: Scalar):
//    %1 Tensor = aten::add(%p0, %p1, %p2)
//  return %1
//
// Execute: CastToTensorInputs(sub_graph)
//
// After:
//  with prim::FusionGroup(
//      %p0.1: Tensor,
//      %p1.1: Tensor,
//      %p2.1: Tensor):
//    %1 int = aten::item(%p2.1)
//    %2 Tensor = aten::add(%p0.1, %p1.1, %1)
//    return %2
void CastBoundaryScalarToTensor(Graph* disc_graph, size_t i,
                                at::ScalarType& typ) {
  auto new_input = disc_graph->insertInput(
      i, c10::string(disc_graph->inputs()[i]->debugName() + ".1"));
  new_input->setType(TensorType::create(typ, c10::nullopt, 0, false));
  auto orig_input = disc_graph->inputs()[i + 1];
  auto item_node = disc_graph->create(aten::item, {new_input});
  // TODO(Yancey1989): supports more types
  item_node->output()->setType(c10::IntType::get());
  disc_graph->appendNode(item_node);
  orig_input->replaceAllUsesWith(item_node->output());
  item_node->moveBefore(item_node->output()->uses()[0].user);
  disc_graph->eraseInput(i + 1);
}

at::ScalarType DerivateScalarType(const Type& typ) {
  if (typ.isSubtypeOf(*c10::IntType::get())) {
    return at::ScalarType::Int;
  } else if (typ.isSubtypeOf(*FloatType::get())) {
    return at::ScalarType::Float;
  } else if (typ.isSubtypeOf(*BoolType::get())) {
    return at::ScalarType::Bool;
  }
  TORCH_CHECK(false, "unsupported input dtype: ", typ.str());
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
      auto scalar_type = DerivateScalarType(*input->type());
      CastBoundaryScalarToTensor(disc_graph, i, scalar_type);
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
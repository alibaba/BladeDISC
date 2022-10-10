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

#include "pytorch_blade/ltc/disc_compiler/passes/disc_fuser.h"

#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/script.h>

#include "pytorch_blade/compiler/jit/fusion.h"
#include "pytorch_blade/compiler/mlir/converters/mhlo_conversion.h"
#include "pytorch_blade/ltc/disc_compiler/passes/graph_fuser.h"

#ifdef TORCH_BLADE_BUILD_WITH_CUDA
#include <c10/cuda/CUDAFunctions.h>
#endif

namespace torch_disc {
namespace compiler {
using namespace ::torch::jit;

c10::TypePtr getScalarTypePtr(at::ScalarType& typ) {
  if (c10::isFloatingType(typ)) {
    return c10::FloatType::get();
  } else if (c10::isIntegralType(typ)) {
    return c10::IntType::get();
  } else if (typ == c10::ScalarType::Bool) {
    return c10::BoolType::get();
  }
  TORCH_CHECK(false, "unsupported scalar type: ", typ);
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
void CastBoundaryScalarToTensor(
    Graph* disc_graph,
    size_t i,
    at::ScalarType& typ) {
  auto new_input = disc_graph->insertInput(
      i, c10::string(disc_graph->inputs()[i]->debugName() + ".1"));
  // scalar device should be on CPU
  new_input->setType(
      TensorType::create(typ, torch::Device(torch::kCPU), 0, false));
  auto orig_input = disc_graph->inputs()[i + 1];
  auto item_node = disc_graph->create(aten::item, {new_input});
  item_node->output()->setType(getScalarTypePtr(typ));
  disc_graph->prependNode(item_node);
  orig_input->replaceAllUsesWith(item_node->output());
  disc_graph->eraseInput(i + 1);
}

bool IsDiscFusable(const torch::jit::Node* node) {
  if (node->kind() == prim::Constant)
    return true;
  if (torch::blade::IsMlirMhloSupported(*node)) {
    for (auto& input : node->inputs()) {
      // input should be Tensor or Scalar with explict type
      if (!input->type()->cast<c10::TensorType>() &&
          c10::tryScalarTypeFromJitType(*input->type()) == c10::nullopt &&
          input->node()->kind() != torch::prim::Constant) {
        return false;
      }
    }

    for (auto& output : node->outputs()) {
      if (!output->type()->cast<c10::TensorType>())
        return false;
    }
    return true;
  }
  return false;
}

void DiscFusion(const std::shared_ptr<Graph>& graph) {
  overrideCanFuseOnCPULegacy(true);
  CustomFuseGraph(
      const_cast<std::shared_ptr<Graph>&>(graph),
      &IsDiscFusable,
      torch::jit::Symbol::fromQualString("prim::FusionGroup"));
  overrideCanFuseOnCPULegacy(false);
  torch::jit::EliminateDeadCode(graph);
  CastingScalarInputToTensor(graph);
}

void CastingScalarInputToTensor(
    const std::shared_ptr<torch::jit::Graph>& graph) {
  for (auto node : graph->nodes()) {
    if (node->kind() == torch::prim::FusionGroup) {
      auto sub_graph = node->g(attr::Subgraph);
      auto inputs = node->inputs();
      for (const auto i : c10::irange(inputs.size())) {
        auto input = inputs[i];
        if (input->type()->cast<c10::TensorType>() == nullptr) {
          if (input->node()->matches("aten::item(Tensor self) -> Scalar")) {
            node->replaceInput(i, input->node()->input());
          } else {
            auto num2tensor = graph->createNumToTensor(input);
            num2tensor->insertAfter(input->node());
            node->replaceInput(i, num2tensor->output());
          }

          auto scalar_type = c10::scalarTypeFromJitType(*input->type());
          CastBoundaryScalarToTensor(sub_graph.get(), i, scalar_type);
        }
      }
    }
  }
}

} //  namespace compiler
} //  namespace torch_disc

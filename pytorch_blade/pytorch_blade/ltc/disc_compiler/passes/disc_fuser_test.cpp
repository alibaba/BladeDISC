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

#include <gtest/gtest.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/custom_class.h>
#include <torch/script.h>

#include "pytorch_blade/ltc/disc_compiler/passes/disc_fuser.h"
#include "pytorch_blade/ltc/disc_compiler/passes/graph_fuser.h"

namespace torch_disc {
namespace compiler {

TEST(TestDiscFusion, TestConstantNode) {
  const std::string graph_str = R"IR(
graph(%p1 : Float(64, 1, 28, 28, strides=[784, 784, 28, 1], requires_grad=0, device=cuda:0)):
  %1 : int[] = prim::Constant[value=[64, 784]]()
  %2 : Float(*, *, requires_grad=0, device=cuda:0) = aten::reshape(%p1, %1)
  return (%2)
)IR";
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph_str, g.get());
  DiscFusion(g);
  torch::jit::EliminateDeadCode(g);
  torch::jit::testing::FileCheck()
      .check("prim::FusionGroup_0")
      ->check("prim::Constant")
      ->check("aten::reshape")
      ->run(*g);
}

TEST(TestDiscFusion, TestScalarInput) {
  const std::string graph_str = R"IR(
graph(%p0 : int,
      %p1 : Float(4, 1, strides=[1, 1], requires_grad=0, device=cuda:0),
      %p2 : Float(4, strides=[1], requires_grad=0, device=cuda:0)):
  %4 : Float(*, *, device=cuda:0) = aten::relu(%p2)
  %3 : Float(*, *, device=cuda:0) = aten::add(%4, %p1, %p0)
  return (%3)
)IR";
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph_str, g.get());
  DiscFusion(g);
  torch::jit::EliminateDeadCode(g);
  torch::jit::testing::FileCheck()
      .check("prim::NumToTensor")
      ->check("prim::FusionGroup_0")
      ->check("aten::relu")
      ->check("aten::item")
      ->check("aten::add")
      ->run(*g);
}

TEST(TestDiscFusion, TestCycle) {
  //
  //    sort
  //   /     \
  //  ↓       ↓
  // relu -> add
  //
  auto graph_string_cycle = R"IR(
graph(%p1 : Float(2, 2, strides=[2, 1], requires_grad=0, device=cuda:0),
      %p3 : Float(2, 2, strides=[2, 1], requires_grad=0, device=cuda:0)):
  %4 : int = prim::Constant[value=-1]()
  %5 : bool = prim::Constant[value=0]()
  %6 : Tensor, %7 : Tensor = aten::sort(%p1, %4, %5)
  %8 : Tensor = aten::relu(%6)
  %9 : Tensor = aten::add(%7, %8, %4)
  return (%9)
)IR";

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph_string_cycle, g.get());
  g->lint();
  torch::jit::EliminateDeadCode(g);
  torch::jit::AliasDb db(g);
  overrideCanFuseOnCPULegacy(true);
  CustomFuseGraph(
      g,
      [](torch::jit::Node* n) {
        if (n->kind() == torch::prim::Param || n->kind() == torch::aten::relu)
          return false;
        for (auto inp : n->inputs()) {
          if (inp->type()->isSubtypeOf(*torch::NumberType::get()) &&
              inp->node()->kind() != torch::prim::Constant)
            return false;
        }
        return true;
      },
      torch::jit::Symbol::fromQualString("prim::FusionGroup"));
  overrideCanFuseOnCPULegacy(false);
  torch::jit::EliminateDeadCode(g);
  torch::jit::testing::FileCheck()
      .check("prim::FusionGroup_0")
      ->check("prim::Constant")
      ->check("aten::sort")
      ->run(*g);
  torch::jit::testing::FileCheck()
      .check("prim::FusionGroup_1")
      ->check("prim::Constant")
      ->check("aten::add")
      ->run(*g);
}

} //  namespace compiler
} //  namespace torch_disc

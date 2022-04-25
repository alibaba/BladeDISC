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
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/script.h>
#include "ltc/disc_compiler/passes/cluster.h"

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
  std::cout << g->toString() << std::endl;
  EXPECT_TRUE(false);
}

} //  namespace compiler
} //  namespace torch_disc
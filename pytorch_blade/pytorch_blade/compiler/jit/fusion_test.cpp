// Copyright 2021 The BladeDISC Authors. All rights reserved.
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
#include <vector>

#include "pytorch_blade/compiler/jit/fusion.h"

using namespace torch::blade;
TEST(SetTensorShapeTest, TestCase) {
  const std::string graph_str = R"IR(
graph(%p1 : Float(64, 1, 28, 28, requires_grad=0, device=cuda:0)):
  %1 : Float(64, 1, 28, 28, requires_grad=0, device=cuda:0) = aten::relu(%p1)
  return (%1)
)IR";
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph_str, g.get());
  std::vector<int64_t> dims = {-1, -1, 28, 28};
  set_tensor_shape(g->inputs()[0], dims);
  auto tensor_type = g->inputs()[0]->type()->cast<c10::TensorType>();
  ASSERT_TRUE(tensor_type != nullptr);
  ASSERT_FALSE(tensor_type->sizes()[0].has_value());
  ASSERT_FALSE(tensor_type->sizes()[1].has_value());
  ASSERT_EQ(tensor_type->sizes()[2].value(), 28);
}

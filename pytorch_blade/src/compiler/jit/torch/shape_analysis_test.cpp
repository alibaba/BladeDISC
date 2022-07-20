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
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/script.h>
#include "compiler/jit/torch/shape_analysis.h"

namespace torch {
namespace blade {

TEST(PropagateInputShapesTest, SimpleUnary) {
  const std::string graph_str = R"IR(
graph(%p1 : Float(*, *, *, device=cuda:0),
      %p2 : Float(768, strides=[1], requires_grad=0, device=cuda:0),
      %p3 : Float(768, strides=[1], requires_grad=0, device=cuda:0)):
  %cst_list : int[] = prim::Constant[value=[768]]()
  %cst_float : float = prim::Constant[value=9.9999999999999998e-13]()
  %1 : Tensor, %2 : Tensor, %3 : Tensor = aten::native_layer_norm(%p1, %cst_list, %p2, %p3, %cst_float)
  %4 : (Tensor, Tensor, Tensor) = prim::TupleConstruct(%1, %2, %3)
  return (%4)
)IR";
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph_str, g.get());
  torch::blade::PropagateInputShapes(g);
  std::cout << g->toString() << std::endl;
  torch::jit::testing::FileCheck()
      .check(
          "%5 : Float(*, *, *, device=cuda:0), %6 : Float(*, *, *, device=cuda:0), %7 : Float(*, *, *, device=cuda:0) = aten::native_layer_norm(%p1, %3, %p2, %p3, %4)")
      ->run(*g);
}
} //  namespace blade
} //  namespace torch
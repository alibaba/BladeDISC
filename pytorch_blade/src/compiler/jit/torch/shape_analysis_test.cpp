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
#if PYTORCH_MAJOR_VERSOIN == 1 && PYTORCH_MINOR_VERSION >= 8
  const std::string graph_str = R"IR(
graph(%p1 : Float(*, *, *, device=cuda:0)):
  %1 : Tensor = aten::relu(%p1)
  return (%1)
)IR";
#elif PYTORCH_MAJOR_VERSION == 1 && PYTORCH_MINOR_VERSION > 6
  const std::string graph_str = R"IR(
graph(%p1 : Float(*, *, *, device=cuda:0)):
  %1 : Tensor = aten::relu(%p1)
  return (%1)
)IR";
#else
  const std::string graph_str = R"IR(
graph(%p1 : Float(*, *, *)):
  %1 : Tensor = aten::relu(%p1)
  return (%1)
)IR";
#endif
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(graph_str, g.get());
  torch::blade::PropagateInputShapes(g);
  std::cout << g->toString() << std::endl;
  torch::jit::testing::FileCheck()
      .check(
#if PYTORCH_MAJOR_VERSION == 1 && PYTORCH_MINOR_VERSION <= 6
          "Float(*, *, *) = aten::relu(%p1)")
#else
          "Float(*, *, *, device=cuda:0) = aten::relu(%p1)")
#endif
      ->run(*g);
}
} //  namespace blade
} //  namespace torch
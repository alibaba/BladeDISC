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
#include "compiler/jit/tool_funcs.h"

using namespace torch::jit;
TEST(GraphMutation, ReturnMultiOutputsWithTuple) {
  const std::string source_ir = R"IR(
    graph(%input):
        %0 : Tensor = aten::sign(%input)
        %1 : Tensor = aten::abs(%input)
        %2 : Tensor = aten::log1p(%1)
        %res : Tensor = aten::mul(%0, %2)
        return (%2, %res)
  )IR";

  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  parseIR(source_ir, graph.get(), vmap);
  graph->dump();
  torch::blade::return_multi_outputs_with_tuple(graph);
  graph->dump();
  torch::jit::testing::FileCheck()
      .check("%5 : (Tensor, Tensor) = prim::TupleConstruct(%3, %4)")
      ->check("return (%5)")
      ->run(*graph);
}

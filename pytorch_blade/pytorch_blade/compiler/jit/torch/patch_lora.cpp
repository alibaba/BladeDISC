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

#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

#include <algorithm>
#include <unordered_map>

namespace torch {
namespace blade {
using namespace ::torch::jit;

bool markLoraInputs(Block* b) {
  int64_t weight_index = 0;
  for (Node* node : b->nodes()) {
    if (node->kind() == aten::dropout &&
        node->input(0)->node()->kind() == aten::add &&
        node->input(0)->node()->input(0)->node()->kind() == aten::mul &&
        node->input(0)->node()->input(1)->node()->kind() == aten::linear) {
      auto to_out_up =
          node->input(0)->node()->input(1)->node()->input(0)->node();
      auto to_out_down = to_out_up->input(0)->node();
      // add down.wight to inputs
      auto down_weights =
          b->addInput("down_weights" + std::to_string(weight_index));
      auto up_weights =
          b->addInput("up_weights" + std::to_string(weight_index));
      to_out_up->replaceInput(1, up_weights);
      to_out_down->replaceInput(1, down_weights);
      weight_index++;
    }
  }
  return true;
}

bool markLoraInputs(Graph* graph) {
  return markLoraInputs(graph->block());
}
} // namespace blade
} // namespace torch

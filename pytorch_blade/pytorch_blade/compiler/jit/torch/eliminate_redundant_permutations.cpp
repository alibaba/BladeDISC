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

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/remove_inplace_ops.h>
#include <stack>
#include "pytorch_blade/compiler/jit/torch/alias_analysis.h"

namespace torch {
namespace blade {
using ClassType = c10::ClassType;
using namespace torch::jit;

namespace {
bool parse_permute_dims(Node* dims, std::vector<size_t>& indices) {
  for (Value* input : dims->inputs()) {
    if (input->node()->kind() != prim::Constant) {
      return false;
    }
    int index = constant_as<int>(input).value_or(-1);
    if (index < 0) {
      return false;
    }
    indices.push_back(index);
  }
  return true;
}

// The example bellow first permutes NHWC to NCHW, and then to NHWC, so skipped
// Before: NodeA --> Perm(0, 3, 1, 2) --> Perm(0, 2, 3, 1) --> NodeB
// After:  NodeA --> NodeB
void eliminate_redundant_permutations(Block* block) {
  auto it = block->nodes().begin();
  while (it != block->nodes().end()) {
    auto node = *it;
    ++it;
    for (auto block : node->blocks()) {
      eliminate_redundant_permutations(block);
    }
    if (node->kind() != aten::permute) {
      continue;
    }
    Node* producer = node->inputs()[0]->node();
    if (producer->kind() != aten::permute) {
      continue;
    }
    Node* consumer_dims = node->inputs()[1]->node();
    if (consumer_dims->kind() != prim::ListConstruct) {
      continue;
    }
    Node* producer_dims = producer->inputs()[1]->node();
    if (producer_dims->kind() != prim::ListConstruct) {
      continue;
    }
    std::vector<size_t> consumer_indices;
    std::vector<size_t> producer_indices;
    if (!parse_permute_dims(consumer_dims, consumer_indices)) {
      continue;
    }
    if (!parse_permute_dims(producer_dims, producer_indices)) {
      continue;
    }
    if (producer_indices.size() != consumer_indices.size()) {
      continue;
    }
    std::vector<size_t> effective_indices;
    for (size_t producer_index : producer_indices) {
      if (producer_index < consumer_indices.size()) {
        effective_indices.push_back(consumer_indices.at(producer_index));
      }
    }
    if (producer_indices.size() != effective_indices.size()) {
      continue;
    }
    bool effective_permutation = false;
    for (size_t i = 0; i < effective_indices.size(); ++i) {
      if (i != effective_indices.at(i)) {
        effective_permutation = true;
        break;
      }
    }
    if (!effective_permutation) {
      // effectively no permutation, so skip the 2 Permute Nodes
      node->outputs()[0]->replaceAllUsesWith(producer->inputs()[0]);
    }
  }
}
} // namespace

void eliminate_redundant_permutations(
    const std::shared_ptr<torch::jit::Graph>& graph) {
  eliminate_redundant_permutations(graph->block());
  EliminateDeadCode(graph->block());
}

} // namespace blade
} // namespace torch

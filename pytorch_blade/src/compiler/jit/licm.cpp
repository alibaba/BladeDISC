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
#include <stack>
#include "compiler/jit/torch/alias_analysis.h"

namespace torch {
namespace blade {
using ClassType = c10::ClassType;
using namespace torch::jit;

namespace {
static const std::unordered_set<NodeKind> whiteList = {
    aten::add,        aten::sub,      aten::squeeze, aten::unsqueeze,
    aten::layer_norm, aten::matmul,   aten::addmm,   aten::conv2d,
    aten::transpose,  aten::softmax,  aten::size,    aten::relu,
    aten::contiguous, aten::floordiv, aten::lt,      aten::gt,
    aten::eq,         aten::le,       aten::ge,      aten::__range_length,
    aten::embedding,  aten::t,        prim::GetAttr};

static std::unordered_set<const Value*> variant_values;
void node_licm(Node* loop) {
  Graph* graph = loop->owningGraph();
  Block* body = loop->blocks().at(0);
  for (auto input : body->inputs()) {
    variant_values.insert(input);
  }
  for (auto it = body->nodes().begin(); it != body->nodes().end();) {
    // licm might destroy the current node, so we need to pre-increment
    // the iterator
    Node* node = *it;
    ++it;
    bool variant = false;
    for (auto input : node->inputs()) {
      if (variant_values.count(input)) {
        variant = true;
        break;
      }
    }
    if (variant) {
      for (auto output : node->outputs()) {
        variant_values.insert(output);
      }
      continue;
    }
    WithInsertPoint insert_point_guard{loop};
    Node* clone =
        graph->insertNode(graph->createClone(node, [](Value* v) { return v; }));
    for (size_t i = 0; i < node->outputs().size(); ++i) {
      node->outputs()[i]->replaceAllUsesWith(clone->outputs()[i]);
    }
    node->destroy();
  }
}

inline bool isForLoop(Node* node) {
  return node->kind() == prim::Loop;
}

void BlockLICM(Block* block) {
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    // licm might insert nodes, so we need to pre-increment
    // the iterator
    Node* node = *it;
    ++it;
    auto schema = node->maybeSchema();
    if (schema && schema->is_mutable() || whiteList.count(node->kind()) == 0) {
      if (node->inputs().size() > 0) {
        // Note(minmin): Assume only arg0 can be mutated. FixMe if any
        // counter-example found in the future.
        variant_values.insert(node->inputs().at(0));
      }
    }
    for (Block* subblock : node->blocks()) {
      BlockLICM(subblock);
    }
    if (isForLoop(node)) {
      node_licm(node);
    }
  }
}
} // namespace

void licm(const std::shared_ptr<torch::jit::Graph>& graph) {
  BlockLICM(graph->block());
  EliminateCommonSubexpression(graph);
  LintGraph(graph);
}

} // namespace blade
} // namespace torch

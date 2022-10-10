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

#include "pytorch_blade/compiler/jit/torch/const_loop_unroll.h"
#include <ATen/core/interned_strings.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
namespace torch {
namespace blade {
using namespace torch::jit;
namespace {
static constexpr int64_t kMaxBodyRepeats = 64;
bool isTrueConstant(Value* val) {
  c10::optional<bool> maybe_value = constant_as<bool>(val);
  return maybe_value && *maybe_value;
}
bool isForLoop(Node* node) {
  if (node->kind() != prim::Loop)
    return false;
  Value* start_cond = node->inputs().at(1);
  Value* continue_cond = node->blocks().at(0)->outputs().at(0);
  return isTrueConstant(start_cond) && isTrueConstant(continue_cond);
}
// XXX: This function can only be called with a loop that is guaranteed to
// execute EXACTLY ONCE.
void inlineBody(Node* loop) {
  auto graph = loop->owningGraph();
  auto body = loop->blocks().at(0);
  WithInsertPoint insert_point_guard{loop};
  std::unordered_map<Value*, Value*> value_map;
  auto get_value = [&](Value* v) {
    auto it = value_map.find(v);
    if (it != value_map.end())
      return it->second;
    return v;
  };
  // Loop node has extra (max_iters, initial_cond) inputs,
  // body has an extra (loop_counter) input.
  for (size_t i = 2; i < loop->inputs().size(); ++i) {
    value_map[body->inputs()[i - 1]] = loop->inputs()[i];
  }
  for (Node* orig : body->nodes()) {
    Node* clone = graph->insertNode(graph->createClone(orig, get_value));
    for (size_t i = 0; i < orig->outputs().size(); ++i) {
      value_map[orig->outputs()[i]] = clone->outputs()[i];
    }
  }
  for (size_t i = 0; i < loop->outputs().size(); ++i) {
    loop->outputs().at(i)->replaceAllUsesWith(
        get_value(body->outputs().at(i + 1)));
  }
  // XXX: it is extremely important to destroy the loop in here. DCE might not
  // be able to conclude that it's safe, because the loop might contain side
  // effects.
  loop->destroy();
}
// inserts a copy of body, passing inputs to the inputs of the block
// it returns the a list of the Values for the output of the block
std::vector<Value*> insertBlockCopy(
    Graph& graph,
    Block* body,
    at::ArrayRef<Value*> inputs) {
  TORCH_INTERNAL_ASSERT(inputs.size() == body->inputs().size());
  std::unordered_map<Value*, Value*> value_map;
  auto get_value = [&](Value* v) {
    auto it = value_map.find(v);
    if (it != value_map.end())
      return it->second;
    return v;
  };
  auto inputs_it = inputs.begin();
  for (Value* input : body->inputs()) {
    value_map[input] = *inputs_it++;
  }
  for (Node* node : body->nodes()) {
    Node* new_node = graph.insertNode(graph.createClone(node, get_value));
    auto outputs_it = new_node->outputs().begin();
    for (Value* output : node->outputs()) {
      value_map[output] = *outputs_it++;
    }
  }
  return fmap(body->outputs(), get_value);
}
void repeatBody(Block* body, size_t times, Block* dest) {
  auto graph = body->owningGraph();
  WithInsertPoint insert_point_guard(dest);
  for (Value* input : body->inputs()) {
    dest->addInput()->copyMetadata(input);
  }
  std::vector<Value*> io = dest->inputs().vec();
  TORCH_INTERNAL_ASSERT(
      !body->inputs().at(0)->hasUses(), "loop counter should be unused");
  for (size_t i = 0; i < times; ++i) {
    io[0] = body->inputs().at(0);
    io = insertBlockCopy(*graph, body, io);
  }
  for (Value* output : io) {
    dest->registerOutput(output);
  }
  // It's likely that we have some dead nodes now - for example the "true"
  // constant that prevents the loop from breaking. We shouldn't wait too long
  // before removing them because they might artificially increase the loop size
  // and prevent outer loop unrolling.
  EliminateDeadCode(dest, false);
}
// Replaces the builtin loop counter with a "mutable" variable outside of the
// loop.
void replaceLoopCounter(Node* loop) {
  Graph* graph = loop->owningGraph();
  Block* body = loop->blocks().at(0);
  WithInsertPoint guard(loop);
  Value* init_counter = graph->insertConstant(0);
  loop->insertInput(2, init_counter);
  loop->insertOutput(0)->setType(IntType::get());
  Value* internal_counter = body->insertInput(1)->setType(init_counter->type());
  body->inputs()[0]->replaceAllUsesWith(internal_counter);
  WithInsertPoint insertPointGuard{body->return_node()};
  Value* result = graph->insert(aten::add, {internal_counter, 1});
  body->insertOutput(1, result);
}
void const_unroll(Node* loop) {
  Graph* graph = loop->owningGraph();
  Block* body = loop->blocks().at(0);
  // We will be using a "mutable" counter outside of the loop instead of the
  // default one, because this will allow us to share it between the unrolled
  // loop and its epilogue. This is necessary only if the loop counter is
  // actually used in the body.
  if (body->inputs()[0]->uses().size() > 0)
    replaceLoopCounter(loop);
  // Some optimization for constant-length loops. If we know they won't run too
  // many times, then we can unroll them entirely.
  Value* trip_count = loop->inputs().at(0);
  c10::optional<int64_t> const_len = constant_as<int64_t>(trip_count);
  if (const_len && *const_len < kMaxBodyRepeats) {
    Block* dest = loop->addBlock();
    repeatBody(body, *const_len, dest);
    loop->eraseBlock(0);
    inlineBody(loop);
    return;
  }
}
void UnrollConstLoops(Block* block) {
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    // XXX: unroll might destroy the current node, so we need to pre-increment
    // the iterator
    Node* node = *it;
    ++it;
    for (Block* subblock : node->blocks()) {
      UnrollConstLoops(subblock);
    }
    if (isForLoop(node)) {
      const_unroll(node);
    }
  }
}
} // anonymous namespace
void UnrollConstLoops(std::shared_ptr<Graph>& graph) {
  UnrollConstLoops(graph->block());
  EliminateDeadCode(graph);
}
} // namespace blade
} // namespace torch

// Copyright 2023 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <memory>

namespace torch {
namespace jit {
struct Graph;
struct Block;
struct Node;
struct Value;
} // namespace jit
} // namespace torch

namespace torch {
namespace blade {

using torch::jit::Block;
using torch::jit::Graph;
using torch::jit::Node;
using torch::jit::Value;

// Runs constant propagation on all objects unless ignore_custom_classes is
// specified as true, in which case user defined classes are skipped.  This is
// useful to prevent early fusion of packing operations, which end up lowering
// away information about their constructors (e.g. packed::linear_clamp_prepack
// and prepacked::conv2d_clamp_prepack)
// Returns True if the pass made a change to the graph
bool ConstantPropagation(std::shared_ptr<Graph>& graph);

// runs constant propagation only on ops that have non-aliasing inputs & outputs
// Returns True if the pass made a change to the graph
// bool ConstantPropagationImmutableTypes(std::shared_ptr<Graph>& graph);

// Runs the node if its inputs are constants. Callers of this function must
// make their own determination if constant prop is appropriate - for example
// non-deterministic ops or ops with side effects.  If ignore_custom_classes is
// specified, nodes that output user defined classes are not run.
// TORCH_API c10::optional<Stack> runNodeIfInputsAreConstant(
//    const Node* node,
//    bool ignore_custom_classes = false,
//    AliasDb* db = nullptr);

} // namespace blade
} // namespace torch

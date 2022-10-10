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

#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch_disc {
namespace compiler {

TORCH_API bool canFuseOnCPULegacy();
TORCH_API void overrideCanFuseOnCPULegacy(bool value);

// NB: Be sure to run DCE before fusion, because dead instructions
// can prevent fusion opportunities from being exploited.
// On Windows will noop, NYI
TORCH_API void FuseGraph(
    std::shared_ptr<torch::jit::Graph>& graph,
    bool strict_fuser_check = false);

// \brief Custom fusion pass using a node-level callback to
// determine the inclusion of nodes in a subgraph.
//
// This helper omits aliased inputs and fusion across control flow
// boundaries.
//
// \arg graph The graph to be modified in-place
// \arg is_fusable A callback run on each fusable node in the graph.
// \arg kind The label given to the resultant fused subgraph
// \arg arg_limit The maximum number of args the resultant fused subgraph
//                should have.  Note: This will likely develop into a general
//                post condition on the fused subgraph.
TORCH_API void CustomFuseGraph(
    std::shared_ptr<torch::jit::Graph>& graph,
    const std::function<bool(torch::jit::Node*)>& is_fusable,
    torch::jit::Symbol kind,
    size_t arg_limit = std::numeric_limits<size_t>::max());

} // namespace compiler
} // namespace torch_disc

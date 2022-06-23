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

#include <memory>
#include <tuple>

namespace torch {
namespace jit {
class Graph;
class Node;
} // namespace jit
} // namespace torch

namespace torch {
namespace blade {

bool IsTorchMlirAvailable();

bool IsMlirMhloSupported(const torch::jit::Node&);

// Return a pair of the MLIR module strings, with the first one in
// parsable/compilable format and the second one in pretty format which elide
// large elements like constant tensors.
std::tuple<std::string, std::string, std::string, std::string>
ConvertTorchScriptToMhlo(std::shared_ptr<torch::jit::Graph> graph);
} // namespace blade
} // namespace torch

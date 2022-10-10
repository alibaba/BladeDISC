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

#include <ATen/core/interned_strings.h>
#include <memory>
#include <unordered_map>
#include <vector>

#if PYTORCH_MAJOR_VERSION == 1 && PYTORCH_MINOR_VERSION >= 12
namespace torch {
namespace jit {
struct OperatorSet;
struct Node;
} // namespace jit

namespace blade {
using ::torch::jit::OperatorSet;
bool nodeIsMemberOf(const torch::jit::Node& node, const OperatorSet& os);
} // namespace blade
} // namespace torch
#else
namespace torch {
namespace jit {
class Operator;
struct Node;
} // namespace jit

namespace blade {
struct OperatorSet {
  OperatorSet(std::initializer_list<const char*> sig_literals);
  std::vector<std::shared_ptr<torch::jit::Operator>> getOps() const;
  void insert(std::initializer_list<const char*> sig_literals);

  bool hasMember(const torch::jit::Node& node) const;

 private:
  friend struct torch::jit::Node;
  std::unordered_map<
      ::c10::Symbol,
      std::vector<std::shared_ptr<torch::jit::Operator>>>
      ops;
};
bool nodeIsMemberOf(const torch::jit::Node& node, const OperatorSet& os);
} // namespace blade
} // namespace torch
#endif

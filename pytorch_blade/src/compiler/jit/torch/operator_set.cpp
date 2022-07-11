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

#include "operator_set.h"
#include <torch/script.h>

namespace torch {
namespace blade {

#if PYTORCH_MAJOR_VERSION == 1 && PYTORCH_MINOR_VERSION >= 12
bool nodeIsMemberOf(const torch::jit::Node& node, const OperatorSet& os) {
  return node.isMemberOf(os);
}
#else
using ::c10::Symbol;
using ::torch::jit::Operator;

OperatorSet::OperatorSet(std::initializer_list<const char*> sig_literals) {
  insert(sig_literals);
}

std::vector<std::shared_ptr<Operator>> OperatorSet::getOps() const {
  std::vector<std::shared_ptr<Operator>> result;
  for (const auto& kv : ops) {
    auto ops_for_symbol = kv.second;
    result.insert(result.end(), ops_for_symbol.begin(), ops_for_symbol.end());
  }
  return result;
}

void OperatorSet::insert(std::initializer_list<const char*> sig_literals) {
  for (const char* sig : sig_literals) {
    auto op = ::torch::jit::getOperatorForLiteral(sig);
    ops[Symbol::fromQualString(op->schema().name())].push_back(op);
  }
}

bool OperatorSet::hasMember(const torch::jit::Node& node) const {
  auto it = ops.find(node.kind());
  if (it == ops.end()) {
    return false;
  }
  for (auto& op : it->second) {
    if (node.matches(op->schema())) {
      return true;
    }
  }
  return false;
}

bool nodeIsMemberOf(const torch::jit::Node& node, const OperatorSet& os) {
  return os.hasMember(node);
}
#endif
} // namespace blade
} // namespace torch

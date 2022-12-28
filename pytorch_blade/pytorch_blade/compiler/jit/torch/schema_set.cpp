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

#include "pytorch_blade/compiler/jit/torch/schema_set.h"
#include <torch/script.h>

namespace torch {
namespace blade {

using ::c10::Symbol;
SchemaSet::SchemaSet(std::initializer_list<const char*> sig_literals) {
  for (const char* sig : sig_literals) {
    auto schema = torch::jit::parseSchema(sig);
    ops[Symbol::fromQualString(schema.name())].push_back(schema);
  }
}

bool SchemaSet::hasMember(const torch::jit::Node& node) const {
  auto it = ops.find(node.kind());
  if (it == ops.end()) {
    return false;
  }
  for (auto& schema : it->second) {
    if (node.matches(schema)) {
      return true;
    }
  }
  return false;
}

bool nodeIsMemberOf(const torch::jit::Node& node, const SchemaSet& os) {
  return os.hasMember(node);
}
} // namespace blade
} // namespace torch

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

#include <torch/csrc/lazy/ts_backend/ts_node.h>

#include <vector>

namespace torch {
namespace lazy {

class TORCH_API Expand : public TsNode {
 public:
  static OpKind ClassOpKind() {
    return OpKind(at::aten::expand);
  }

  Expand(const Value& input, std::vector<int64_t> size, bool is_scalar_expand);

  std::string ToString() const override;

  const std::vector<int64_t>& size() const {
    return size_;
  }

  bool is_scalar_expand() const {
    return is_scalar_expand_;
  }

 private:
  std::vector<int64_t> size_;
  // True iff the input was a scalar and this was generated internally by a
  // lowering and not by user action. For some backends, this difference can be
  // material (for example setting strides according to eager semantics).
  bool is_scalar_expand_;
};

} // namespace lazy
} // namespace torch

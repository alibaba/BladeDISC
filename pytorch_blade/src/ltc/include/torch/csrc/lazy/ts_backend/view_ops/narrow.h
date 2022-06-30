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

namespace torch {
namespace lazy {

class TORCH_API Narrow : public TsNode {
 public:
  static OpKind ClassOpKind() {
    return OpKind(at::aten::narrow);
  }

  Narrow(
      const Value& input,
      c10::ArrayRef<int64_t> base_indices,
      c10::ArrayRef<int64_t> sizes);

  std::string ToString() const override;

  const std::vector<int64_t>& base_indices() const {
    return base_indices_;
  }

  const std::vector<int64_t>& sizes() const {
    return sizes_;
  }

 private:
  std::vector<int64_t> base_indices_;
  std::vector<int64_t> sizes_;
};

} // namespace lazy
} // namespace torch

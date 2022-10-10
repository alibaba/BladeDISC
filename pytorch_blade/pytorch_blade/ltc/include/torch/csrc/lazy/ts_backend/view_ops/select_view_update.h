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

#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>
#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch {
namespace lazy {

class TORCH_API SelectViewUpdate : public TsNode {
 public:
  static OpKind ClassOpKind() {
    return ltc_select_view_update;
  }

  SelectViewUpdate(
      const Value& target,
      const Value& source,
      int64_t dim,
      int64_t start,
      int64_t end,
      int64_t stride);

  std::string ToString() const override;

  int64_t dim() const {
    return dim_;
  }

  int64_t start() const {
    return start_;
  }

  int64_t end() const {
    return end_;
  }

  int64_t stride() const {
    return stride_;
  }

 private:
  int64_t dim_;
  int64_t start_;
  int64_t end_;
  int64_t stride_;
};

} // namespace lazy
} // namespace torch

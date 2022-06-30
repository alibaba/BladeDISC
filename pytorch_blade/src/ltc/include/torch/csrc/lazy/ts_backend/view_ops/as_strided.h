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

class TORCH_API AsStrided : public TsNode {
 public:
  static OpKind ClassOpKind() {
    return OpKind(at::aten::as_strided);
  }

  AsStrided(
      const Value& input,
      std::vector<int64_t> size,
      std::vector<int64_t> stride,
      int64_t storage_offset);

  std::string ToString() const override;

  const std::vector<int64_t>& size() const {
    return size_;
  }

  const std::vector<int64_t>& stride() const {
    return stride_;
  }

  int64_t storage_offset() const {
    return storage_offset_;
  }

  static bool StrideIsSupported(c10::ArrayRef<int64_t> stride);

  static std::vector<int64_t> GetArrayStridePermutation(
      c10::ArrayRef<int64_t> stride);

 private:
  std::vector<int64_t> size_;
  std::vector<int64_t> stride_;
  int64_t storage_offset_;
};

} // namespace lazy
} // namespace torch

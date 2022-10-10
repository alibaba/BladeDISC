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

#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>
#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch {
namespace lazy {

class TORCH_API Cast : public TsNode {
 public:
  static OpKind ClassOpKind() {
    return ltc_cast;
  }

  Cast(
      const Value& input,
      at::ScalarType dtype,
      c10::optional<at::ScalarType> stype = c10::nullopt);

  std::string ToString() const override;

  at::ScalarType dtype() const {
    return dtype_;
  }

  const c10::optional<at::ScalarType>& stype() const {
    return stype_;
  }

 private:
  at::ScalarType dtype_;
  c10::optional<at::ScalarType> stype_;
};

} // namespace lazy
} // namespace torch

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

#include <c10/core/Scalar.h>
#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch {
namespace lazy {

// Differently from Constant, this is a scalar value broadcasted to a shape.
// Even though a Constant could have been used, for simple scalars broadcasted
// to big shapes, the Constant leads to big literals expanded within the
// computation graph.
class TORCH_API Scalar : public TsNode {
 public:
  static OpKind ClassOpKind() {
    return OpKind(at::prim::Constant);
  }

  Scalar(const at::Scalar& value, Shape shape);
  Scalar(const at::Scalar& value, c10::ScalarType type);

  std::string ToString() const override;

  const at::Scalar& value() const {
    return value_;
  }

 private:
  at::Scalar value_;
};

TORCH_API hash_t ScalarHash(const at::Scalar& s);

} // namespace lazy
} // namespace torch

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

#include <torch/csrc/lazy/core/ir_builder.h>

namespace torch {
namespace lazy {

// Generic IR Node implementation for nodes which can simply be described by a
// specific OpKind and a lowering function. IR nodes carrying
// metadata should not be using this class TORCH_API (and have the metadata
// captured by the LowerFn), but they should instead create a dedicated IR node.
// Doing the former would limit IR introspection.
class TORCH_API Generic : public TsNode {
 public:
  Generic(
      OpKind op,
      OpList operands,
      Shape shape,
      size_t num_outputs = 1,
      hash_t hash_seed = static_cast<uint32_t>(0x5a2d296e9));

  Generic(
      OpKind op,
      OpList operands,
      const std::function<Shape()>& shape_fn,
      size_t num_outputs = 1,
      hash_t hash_seed = static_cast<uint32_t>(0x5a2d296e9));

  Generic(
      OpKind op,
      OpList operands,
      size_t num_outputs = 1,
      hash_t hash_seed = static_cast<uint32_t>(0x5a2d296e9));

  Generic(OpKind op, Shape shape, size_t num_outputs, hash_t hash_seed);

 private:
  hash_t hash_seed_;
};

inline NodePtr GenericOp(
    OpKind op,
    OpList operands,
    Shape shape,
    size_t num_outputs = 1,
    hash_t hash_seed = static_cast<uint32_t>(0x5a2d296e9)) {
  return MakeNode<Generic>(
      op, operands, std::move(shape), num_outputs, hash_seed);
}

} // namespace lazy
} // namespace torch

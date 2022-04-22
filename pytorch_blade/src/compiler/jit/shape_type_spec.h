// Copyright 2021 The BladeDISC Authors. All rights reserved.
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

#include <torch/script.h>
#include <functional>
#include <string>
#include <vector>

namespace torch {
namespace blade {
const char* ScalarTypeToString(ScalarType t);
c10::optional<at::ScalarType> ScalarTypeFromString(const std::string& dtype);

struct ShapeType {
  std::vector<int64_t> shape;
  at::ScalarType type;
  bool operator==(const ShapeType& rhs) const {
    return shape == rhs.shape && type == rhs.type;
  }
  std::string Serialize() const;
  static ShapeType Deserialize(const std::string&);
};

class ShapeTypeSpec {
 public:
  ShapeTypeSpec(std::vector<ShapeType> shape_types);
  bool operator==(const ShapeTypeSpec& rhs) const {
    if (hash_code_ != rhs.hash_code_) {
      return false;
    }
    return shape_types_ == rhs.shape_types_;
  }
  size_t hashCode() const {
    return hash_code_;
  }

  std::string Serialize() const;
  static ShapeTypeSpec Deserialize(const std::string&);
  static ShapeTypeSpec GetShapeTypeSpec(const at::List<at::Tensor>&);
  static ShapeTypeSpec GetShapeTypeSpec(
      const std::vector<const torch::jit::Value*>& values);

  const std::vector<ShapeType>& shape_types() const {
    return shape_types_;
  }

 private:
  void AddShapeType(const ShapeType& shape_type);
  size_t hash_code_;
  std::vector<ShapeType> shape_types_;
};

} // namespace blade
} // namespace torch
namespace std {
template <>
struct hash<torch::blade::ShapeTypeSpec> {
  std::size_t operator()(torch::blade::ShapeTypeSpec const& s) const noexcept {
    return s.hashCode();
  }
};
} // namespace std

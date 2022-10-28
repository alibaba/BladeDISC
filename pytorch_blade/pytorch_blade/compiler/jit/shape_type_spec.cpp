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

#include "pytorch_blade/compiler/jit/shape_type_spec.h"

#include <iterator>
#include <regex>

#include <c10/core/ScalarType.h>

#include "pytorch_blade/common_utils/utils.h"
#include "pytorch_blade/compiler/jit/tool_funcs.h"

namespace torch {
namespace blade {
namespace {
inline size_t hash_combine(size_t seed, size_t value) {
  return seed ^ (value + 0x9e3779b9 + (seed << 6u) + (seed >> 2u));
}
} // namespace

const char* ScalarTypeToString(ScalarType t) {
#define DEFINE_CASE(_, name) \
  case ScalarType::name:     \
    return #name;

  switch (t) {
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_CASE)
    default:
      return "UNKNOWN_SCALAR";
  }
#undef DEFINE_CASE
}

c10::optional<at::ScalarType> ScalarTypeFromString(const std::string& dtype) {
#define DEFINE_SCALAR_TYPE(_, name) {#name, at::ScalarType::name},

  static std::unordered_map<std::string, at::ScalarType> type_map = {
      AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_SCALAR_TYPE)};

  auto type = type_map.find(dtype);
  if (type != type_map.end()) {
    return type->second;
  }
  return c10::nullopt;
#undef DEFINE_SCALAR_TYPE
}

std::string ShapeType::Serialize() const {
  std::stringstream repr_str;
  repr_str << ScalarTypeToString(type) << "(";
  std::transform(
      shape.begin(),
      shape.end(),
      std::ostream_iterator<std::string>(repr_str, ","),
      [](int64_t dim) -> std::string {
        return std::to_string(dim) + ":" + std::to_string(dim);
      });
  repr_str << ")";
  return repr_str.str();
}

ShapeType ShapeType::Deserialize(const std::string& serial_str) {
  // something like: "Float(1:1,2:2,3:3,4:4,)"
  const std::regex shp_regex("([a-zA-Z]+)\\((([0-9]+:[0-9]+,)*)\\)");
  std::smatch pieces_match;
  TORCH_CHECK(
      std::regex_match(serial_str, pieces_match, shp_regex),
      serial_str,
      " is illegal.")
  TORCH_CHECK(pieces_match.size() == 4);
  ShapeType shape_type;
  std::string dtype = pieces_match[1].str();
  auto scalar_type = ScalarTypeFromString(dtype);
  TORCH_CHECK(scalar_type, dtype, " is not supported");
  shape_type.type = *scalar_type;

  std::string shape = pieces_match[2].str();
  if (shape.empty()) {
    return shape_type;
  }
  std::vector<std::string> dim_strs = split(shape, ",");
  shape_type.shape.reserve(dim_strs.size());
  std::transform(
      dim_strs.begin(),
      dim_strs.end(),
      std::back_inserter(shape_type.shape),
      [](const std::string& s) -> int64_t {
        // string like: 1:1 (min:max)
        CHECK(!s.empty());
        std::vector<std::string> dim_range = split(s, ":");
        CHECK(dim_range.size() == 2);
        int64_t dim_min = c10::stoll(dim_range[0]);
        int64_t dim_max = c10::stoll(dim_range[1]);
        // TODO: to support a [min, max] range, where min!=max
        CHECK(dim_min == dim_max);
        return dim_min;
      });
  return shape_type;
}

ShapeTypeSpec::ShapeTypeSpec(std::vector<ShapeType> shape_types)
    : shape_types_(std::move(shape_types)), hash_code_(0) {
  for (const auto& shape_type : shape_types_) {
    AddShapeType(shape_type);
  }
}

void ShapeTypeSpec::AddShapeType(const ShapeType& shape_type) {
  const auto& dim = shape_type.shape.size();
  hash_code_ = hash_combine(hash_code_, dim);
  for (size_t k = 0; k < shape_type.shape.size(); ++k) {
    int64_t dim_size = shape_type.shape[k];
    hash_code_ = hash_combine(hash_code_, std::hash<int64_t>{}(dim_size));
  }
  size_t type_code = (size_t)shape_type.type;
  hash_code_ = hash_combine(hash_code_, type_code);
}

std::string ShapeTypeSpec::Serialize() const {
  std::stringstream str_ss;
  for (const auto& shape_type : shape_types_) {
    str_ss << shape_type.Serialize() << ";";
  }
  return str_ss.str();
}

ShapeTypeSpec ShapeTypeSpec::Deserialize(const std::string& serial_str) {
  std::vector<std::string> tokens = split(serial_str, ";");
  std::vector<ShapeType> shape_types;
  shape_types.reserve(tokens.size());
  for (const auto& t : tokens) {
    TORCH_CHECK(!t.empty(), t, "is not a shape string.");
    shape_types.emplace_back(std::move(ShapeType::Deserialize(t)));
  }
  return ShapeTypeSpec(std::move(shape_types));
}

ShapeTypeSpec ShapeTypeSpec::GetShapeTypeSpec(
    const at::List<at::Tensor>& inputs) {
  std::vector<ShapeType> shape_types(inputs.size());
  for (size_t k = 0; k < inputs.size(); ++k) {
    auto& shape_type = shape_types[k];
    at::Tensor inp_tensor = inputs[k];
    const auto& inp_shape = inp_tensor.sizes();
    std::copy(
        inp_shape.cbegin(),
        inp_shape.cend(),
        std::back_inserter(shape_type.shape));
    shape_type.type = inp_tensor.scalar_type();
  }
  return ShapeTypeSpec(shape_types);
}

ShapeTypeSpec ShapeTypeSpec::GetShapeTypeSpec(
    const std::vector<const torch::jit::Value*>& values) {
  std::vector<ShapeType> shape_types;
  shape_types.reserve(values.size());
  for (const auto& val : values) {
    CHECK(is_concrete_shape_tensor_type(*val));
    auto type = val->type()->cast<TensorType>();
    ShapeType shape_type;
    shape_type.shape = *(type->sizes().concrete_sizes());
    shape_type.type = *(type->scalarType());
    shape_types.emplace_back(shape_type);
  }

  return ShapeTypeSpec(shape_types);
}

} // namespace blade
} // namespace torch

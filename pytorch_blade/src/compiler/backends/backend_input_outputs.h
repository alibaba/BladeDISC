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

#include <ATen/core/ScalarType.h>
#include <vector>
namespace torch {
namespace jit {
class Value;
}
} // namespace torch

namespace torch {
namespace blade {
namespace backends {

#define TORCH_BLADE_BACKENDS_DEFINE_FIELD(field, type) \
  void set_##field(const type& val) {                  \
    field = val;                                       \
  }                                                    \
  const type& get_##field() const {                    \
    return field;                                      \
  }                                                    \
  type field

struct DynamicRanges {
  using ShapesType = std::vector<std::vector<int64_t>>;
  using SerialType =
      std::tuple<ShapesType, ShapesType, ShapesType, std::vector<ShapesType>>;

  TORCH_BLADE_BACKENDS_DEFINE_FIELD(min_shape, ShapesType);
  TORCH_BLADE_BACKENDS_DEFINE_FIELD(max_shape, ShapesType);

  // dynamic dimension should be set to -1 when parse onnx to trt network.
  // we store this information in dynamic_setting.
  TORCH_BLADE_BACKENDS_DEFINE_FIELD(dynamic_setting, ShapesType);
  TORCH_BLADE_BACKENDS_DEFINE_FIELD(opt_shapes, std::vector<ShapesType>);

  SerialType Serialize() const;
  static DynamicRanges Deserialize(const SerialType& serialized);
  bool Validate(int);
  bool InRange(const ShapesType& shapes);
  std::string GetShapeString() const;
};

struct TensorInfo {
  TORCH_BLADE_BACKENDS_DEFINE_FIELD(name, std::string);
  TORCH_BLADE_BACKENDS_DEFINE_FIELD(device, std::string);
  TORCH_BLADE_BACKENDS_DEFINE_FIELD(sizes, std::vector<int64_t>);
  c10::ScalarType scalar_type;
  std::string GetDType() const;
  void SetDType(const std::string& dtype);

  using SerialType =
      std::tuple<std::string, std::string, std::string, std::vector<int64_t>>;
  SerialType Serialize() const;
  static TensorInfo Deserialize(const SerialType& serialized);
  TensorInfo() {}
  TensorInfo(const torch::jit::Value& val);
};
} // namespace backends
} // namespace blade
} // namespace torch

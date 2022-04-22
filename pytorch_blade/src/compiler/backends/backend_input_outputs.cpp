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

#include "backend_input_outputs.h"
#include "compiler/jit/shape_type_spec.h"
#include "compiler/jit/tool_funcs.h"

namespace torch {
namespace blade {
namespace backends {

bool DynamicRanges::InRange(const ShapesType& shapes) {
  size_t inp_nums = shapes.size();
  if (inp_nums != min_shape.size() || inp_nums != max_shape.size()) {
    return false;
  }

  for (int k = 0; k < inp_nums; ++k) {
    const auto& min_dims = min_shape[k];
    const auto& max_dims = max_shape[k];
    const auto& dims = shapes[k];
    if (dims.size() != min_dims.size() || dims.size() != max_dims.size()) {
      return false;
    }
    bool in_range = (min_dims <= dims) && (dims <= max_dims);
    if (!in_range) {
      return false;
    }
  }
  return true;
}

bool DynamicRanges::Validate(int inp_nums) {
  // if one the min_shape/max_shape/opt_shapes is empty, we can not construct
  // a valid optimization profile.
  if (min_shape.size() == 0 || max_shape.size() == 0 ||
      opt_shapes.size() == 0) {
    return false;
  }

  // length of min_shape/max_shape/each opt_shape should equal to that of trt
  // network
  // TODO: figure out whether the change of trt input numbers occurred in
  // parsing onnx or building engine. If it is occurred in parsing onnx, this
  // check should be relaxed
  if (min_shape.size() != inp_nums || max_shape.size() != inp_nums) {
    return false;
  }

  for (auto opt_shape : opt_shapes) {
    if (!InRange(opt_shape)) {
      return false;
    }
    for (size_t k = 0; k < opt_shape.size(); ++k) {
      bool in_range =
          (min_shape[k] <= opt_shape[k]) && (opt_shape[k] <= max_shape[k]);
      if (!in_range) {
        return false;
      }
    }
  }
  return true;
}

std::string DynamicRanges::GetShapeString() const {
  auto generateOneTensorStr = [](std::vector<int64_t> t) -> std::string {
    std::stringstream ss;
    std::copy(t.begin(), t.end(), std::ostream_iterator<int64_t>(ss, ", "));
    std::string s_tensor = ss.str();
    s_tensor = s_tensor.substr(0, s_tensor.length() - 2);
    s_tensor = "[" + s_tensor + "]";
    return s_tensor;
  };
  auto generateOneInputStr = [&](ShapesType s) -> std::string {
    std::vector<std::string> temp;
    std::for_each(s.begin(), s.end(), [&](std::vector<int64_t> t) {
      temp.push_back(generateOneTensorStr(t));
    });
    std::stringstream ss;
    std::copy(
        temp.begin(), temp.end(), std::ostream_iterator<std::string>(ss, ", "));
    std::string s_input = ss.str();
    s_input = s_input.substr(0, s_input.length() - 2);
    s_input = "[" + s_input + "]";
    return s_input;
  };
  std::string s;
  s += "min_shape:\n";
  s += generateOneInputStr(min_shape);
  s += "\n";
  s += "max_shape:\n";
  s += generateOneInputStr(max_shape);
  s += "\n";
  s += "opt_shapes:\n";
  for (auto opt_s : opt_shapes) {
    s += generateOneInputStr(opt_s);
    s += "\n";
  }
  s = s.substr(0, s.length() - 1);
  return s;
}

DynamicRanges::SerialType DynamicRanges::Serialize() const {
  return std::make_tuple(min_shape, max_shape, dynamic_setting, opt_shapes);
}

DynamicRanges DynamicRanges::Deserialize(const SerialType& serialized) {
  DynamicRanges ranges;
  ranges.min_shape = std::move(std::get<0>(serialized));
  ranges.max_shape = std::move(std::get<1>(serialized));
  ranges.dynamic_setting = std::move(std::get<2>(serialized));
  ranges.opt_shapes = std::move(std::get<3>(serialized));
  return ranges;
}

TensorInfo::SerialType TensorInfo::Serialize() const {
  return std::make_tuple(name, device, GetDType(), sizes);
}

TensorInfo TensorInfo::Deserialize(const SerialType& serialized) {
  TensorInfo data;
  data.name = std::move(std::get<0>(serialized));
  data.device = std::move(std::get<1>(serialized));
  data.SetDType(std::move(std::get<2>(serialized)));
  data.sizes = std::move(std::get<3>(serialized));
  return data;
}

std::string TensorInfo::GetDType() const {
  return ScalarTypeToString(scalar_type);
}

void TensorInfo::SetDType(const std::string& dtype) {
  auto optional = ScalarTypeFromString(dtype);
  TORCH_CHECK(optional, "The scalarType ", dtype, " is invalid.");
  scalar_type = *optional;
}

TensorInfo::TensorInfo(const torch::jit::Value& val) {
  // default to device cpu
  name = val.debugName();
  device = is_gpu_tensor_type(val) ? "cuda" : "cpu";
  auto tensor_type = val.type()->cast<at::TensorType>();
  TORCH_CHECK(tensor_type != nullptr);
  auto concrete_sizes = tensor_type->sizes().concrete_sizes();
  if (concrete_sizes) {
    sizes = *concrete_sizes;
  }

  if (tensor_type->scalarType()) {
    scalar_type = *(tensor_type->scalarType());
  }
}

} // namespace backends
} // namespace blade
} // namespace torch

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

#include <torch/custom_class.h>

#include <tuple>
#include <vector>

namespace torch {
namespace blade {
namespace quantization {

class FakeQuant : public torch::CustomClassHolder {
 public:
  using AxisType = std::vector<int64_t>;
  using SerialType =
      std::tuple<int64_t, int64_t, int64_t, AxisType, bool, bool, bool, bool>;

  // Construct from serialized data.
  FakeQuant(SerialType serial);
  SerialType serialize() const;
  // scale & zero_point: scalar for per-tensor quant, vector for per-channel
  // quant.
  at::Tensor forward(
      const at::Tensor& input,
      const at::Tensor& scale,
      const at::Tensor& zero_point);

  // accessors
  int64_t quant_min() const {
    return quant_min_;
  }
  int64_t quant_max() const {
    return quant_max_;
  }
  int64_t num_bits() const {
    return num_bits_;
  }
  const AxisType& axis() const {
    return axis_;
  }
  bool isSigned() const {
    return signed_;
  }
  bool isSymmetric() const {
    return symmetric_;
  }
  bool isDynamic() const {
    return dynamic_;
  }
  bool isPerChannel() const {
    return per_channel_;
  }

 private:
  // min value after quantization.
  int64_t quant_min_;
  // max value after quantization.
  int64_t quant_max_;
  // bits number of quantized datatype.
  int64_t num_bits_;
  // refer to a dimension of input for per-channel quantization
  AxisType axis_;
  // int8 or uint8 quantization
  bool signed_;
  // symmetric or asymmetric quantization
  bool symmetric_;
  // true for dynamic quantization.
  bool dynamic_;
  // true for per-channel, false for per-tensor.
  bool per_channel_;
};

} // namespace quantization
} // namespace blade
} // namespace torch

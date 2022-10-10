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

#include <cstdint>
#include <mutex>

#include <ATen/Functions.h>
#include "pytorch_blade/common_utils/logging.h"

#include <torch/script.h>

namespace torch {
namespace blade {
namespace quantization {

// A custom torch op used to carry fake quant info downt to DISC compiler.
torch::Tensor torch_blade_fake_quant(
    torch::Tensor input,
    torch::Tensor scale,
    torch::Tensor zero_point,
    int64_t quant_min,
    int64_t quant_max,
    int64_t num_bits,
    std::vector<int64_t> axis,
    bool use_signed,
    bool use_symmetric,
    bool use_dynamic,
    bool use_per_channel) {
  if (use_per_channel) {
    TORCH_CHECK(
        axis.size() > 0, "Per-channel quantization requires non-empty axis.");
  } else {
    TORCH_CHECK(
        axis.size() == 0, "Per-tensor quantization requires empty axis.");
  }

  if (use_per_channel) {
    // torch version 1.8.x and 1.9.x require Long type for zero_point, while
    // higher versions require Int/Float/Double typle.
#if PYTORCH_MAJOR_VERSION == 0 || \
    PYTORCH_MAJOR_VERSION == 1 && PYTORCH_MINOR_VERSION < 10
    at::Tensor zero_point_new = zero_point.to(at::ScalarType::Long);
#else
    at::Tensor zero_point_new = zero_point.to(at::ScalarType::Int);
#endif
    return at::fake_quantize_per_channel_affine(
        input, scale, zero_point_new, axis[0], quant_min, quant_max);
  } else {
    // use scalar version for backward compatibility.
    return at::fake_quantize_per_tensor_affine(
        input,
        scale.item<double>(),
        zero_point.item<int64_t>(),
        quant_min,
        quant_max);
  }
}

TORCH_LIBRARY(torch_blade, m) {
  m.def("fake_quant", torch_blade_fake_quant);
}

} // namespace quantization
} // namespace blade
} // namespace torch

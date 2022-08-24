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

#include <ATen/Functions.h>

#include <mutex>

#include "common_utils/logging.h"
#include "quantization/fake_quant.h"

namespace torch {
namespace blade {
namespace quantization {

FakeQuant::FakeQuant(FakeQuant::SerialType serial) {
  std::tie(
      quant_min_,
      quant_max_,
      num_bits_,
      axis_,
      signed_,
      symmetric_,
      dynamic_,
      per_channel_) = serial;
  if (per_channel_) {
    TORCH_CHECK(
        axis_.size() > 0, "Per-channel quantization requires non-empty axis.");
  } else {
    TORCH_CHECK(
        axis_.size() == 0, "Per-tensor quantization requires empty axis.");
  }
}

FakeQuant::SerialType FakeQuant::serialize() const {
  return std::make_tuple(
      quant_min_,
      quant_max_,
      num_bits_,
      axis_,
      signed_,
      symmetric_,
      dynamic_,
      per_channel_);
}

at::Tensor FakeQuant::forward(
    const at::Tensor& input,
    const at::Tensor& scale,
    const at::Tensor& zero_point) {
  if (per_channel_) {
    // torch version 1.8.x and 1.9.x require Long type for zero_point, while
    // higher versions require Int/Float/Double typle.
#if PYTORCH_MAJOR_VERSION == 1 && PYTORCH_MINOR_VERSION < 10
    at::Tensor zero_point_new = zero_point.to(at::ScalarType::Long);
#else
    at::Tensor zero_point_new = zero_point.to(at::ScalarType::Int);
#endif
    return at::fake_quantize_per_channel_affine(
        input, scale, zero_point_new, axis_[0], quant_min_, quant_max_);
  } else {
    // use scalar version for backward compatibility.
    return at::fake_quantize_per_tensor_affine(
        input,
        scale.item<double>(),
        zero_point.item<int64_t>(),
        quant_min_,
        quant_max_);
  }
}

namespace {
auto reg =
    torch::class_<FakeQuant>("torch_blade", "FakeQuant")
        .def(torch::init<FakeQuant::SerialType>())
        .def(
            "forward",
            [](const c10::intrusive_ptr<FakeQuant>& self,
               const at::Tensor& input,
               const at::Tensor& scale,
               const at::Tensor& zero_point) {
              return self->forward(input, scale, zero_point);
            })
        .def("quant_min", &FakeQuant::quant_min)
        .def("quant_max", &FakeQuant::quant_max)
        .def("num_bits", &FakeQuant::num_bits)
        .def("axis", &FakeQuant::axis)
        .def("is_signed", &FakeQuant::isSigned)
        .def("is_symmetric", &FakeQuant::isSymmetric)
        .def("is_dynamic", &FakeQuant::isDynamic)
        .def("is_per_channel", &FakeQuant::isPerChannel)
        .def_pickle(
            [](const c10::intrusive_ptr<FakeQuant>& self)
                -> FakeQuant::SerialType { return self->serialize(); },
            [](FakeQuant::SerialType serial) -> c10::intrusive_ptr<FakeQuant> {
              return c10::make_intrusive<FakeQuant>(serial);
            });
} // namespace

} // namespace quantization
} // namespace blade
} // namespace torch

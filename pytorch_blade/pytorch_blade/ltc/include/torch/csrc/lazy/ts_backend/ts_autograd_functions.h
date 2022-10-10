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

#include <torch/csrc/autograd/custom_function.h>

namespace torch {
namespace lazy {

struct MaxPool3dAutogradFunctionTS
    : public torch::autograd::Function<MaxPool3dAutogradFunctionTS> {
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      at::Tensor self,
      at::IntArrayRef kernel_size,
      at::IntArrayRef stride,
      at::IntArrayRef padding,
      at::IntArrayRef dilation,
      bool ceil_mode);
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output);
};

} // namespace lazy
} // namespace torch

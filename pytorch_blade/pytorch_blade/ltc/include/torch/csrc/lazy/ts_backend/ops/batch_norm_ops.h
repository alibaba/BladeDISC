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

namespace torch {
namespace lazy {

// Node for the backward batch norm operator.
class TSNativeBatchNormBackward : public torch::lazy::TsNode {
 public:
  static OpKind ClassOpKind() {
    return OpKind(at::aten::native_batch_norm_backward);
  }

  TSNativeBatchNormBackward(
      const torch::lazy::Value& grad_out,
      const torch::lazy::Value& input,
      const torch::lazy::Value& weight,
      const torch::lazy::Value& running_mean,
      const torch::lazy::Value& running_var,
      const torch::lazy::Value& save_mean,
      const torch::lazy::Value& save_invstd,
      bool training,
      double eps,
      std::array<bool, 3> output_mask);

  TSNativeBatchNormBackward(
      const torch::lazy::Value& grad_out,
      const torch::lazy::Value& input,
      const torch::lazy::Value& weight,
      const torch::lazy::Value& save_mean,
      const torch::lazy::Value& save_invstd,
      bool training,
      double eps,
      std::array<bool, 3> output_mask);

  std::string ToString() const override;

  bool training() const {
    return training_;
  }

  double eps() const {
    return eps_;
  }

  const std::array<bool, 3>& output_mask() const {
    return output_mask_;
  }

 private:
  bool training_;
  double eps_;
  std::array<bool, 3> output_mask_;
};

class TSNativeBatchNormForward : public torch::lazy::TsNode {
 public:
  static OpKind ClassOpKind() {
    return OpKind(at::aten::native_batch_norm);
  }

  TSNativeBatchNormForward(
      const torch::lazy::Value& input,
      const torch::lazy::Value& weight,
      const torch::lazy::Value& bias,
      const torch::lazy::Value& running_mean,
      const torch::lazy::Value& running_var,
      bool training,
      double momentum,
      double eps);

  std::string ToString() const override;

  bool training() const {
    return training_;
  }

  double momentum() const {
    return momentum_;
  }

  double eps() const {
    return eps_;
  }

 private:
  bool training_;
  double momentum_;
  double eps_;
};

} // namespace lazy
} // namespace torch

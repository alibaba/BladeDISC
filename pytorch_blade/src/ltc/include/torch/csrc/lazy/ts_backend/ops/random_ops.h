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

class Normal : public torch::lazy::TsNode {
 public:
  static OpKind ClassOpKind() {
    return OpKind::Get("aten::normal_");
  }

  Normal(
      const torch::lazy::Value& self,
      const double& mean,
      const double& std,
      std::vector<torch::lazy::Shape>&& shapes);

  std::string ToString() const override;
  torch::lazy::TSOpVector Lower(
      std::shared_ptr<torch::jit::GraphFunction> function,
      torch::lazy::TSLoweringContext* loctx) const override;

  double mean_;
  double std_;
};

} // namespace lazy
} // namespace torch

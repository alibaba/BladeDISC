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

#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>
#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch {
namespace lazy {

class TORCH_API DeviceData : public TsNode {
 public:
  static OpKind ClassOpKind() {
    return ltc_device_data;
  }

  explicit DeviceData(std::shared_ptr<BackendData> data);

  // A DeviceData node can be reused if the shape matches,
  // but we will substitute the actual data_ pointer under
  // the hood.
  bool CanBeReused(std::shared_ptr<BackendData> data) const {
    return data_->shape() == data->shape();
  }

  std::string ToString() const override;

  const std::shared_ptr<BackendData>& data() const {
    return data_;
  }

  void SetData(std::shared_ptr<BackendData> data) {
    data_ = data;
  }

  static const DeviceData* Cast(const Node* node);

  // To reuse IR nodes, use this method to create DeviceData nodes
  // instead of calling the constructor directly.
  static NodePtr Create(std::shared_ptr<BackendData> data);

 private:
  std::shared_ptr<BackendData> data_;
};

} // namespace lazy
} // namespace torch

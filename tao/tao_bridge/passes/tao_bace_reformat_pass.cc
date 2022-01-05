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

#define EIGEN_USE_THREADS
#include "tao_bridge/passes/tao_bace_reformat_pass.h"

#include <algorithm>
#include <array>

#include "tao_bridge/common.h"
#include "tensorflow/core/framework/tensor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace tao {

TaoBaCEReformatPass::TaoBaCEReformatPass() {
  auto *bridge_opt = GetTaoBridgeOptions();
  enabled_ = bridge_opt->bace_reformat_enabled;
  max_dim_bar_ = bridge_opt->bace_reformat_max_dim_bar;
  min_dim_bar_ = bridge_opt->bace_reformat_min_dim_bar;
  size_bar_ = bridge_opt->bace_reformat_size_bar;
}

Status TaoBaCEReformatPass::Run(const GraphOptimizationPassOptions &options) {
  if (!enabled_) {
    VLOG(1) << "TaoBaCEReformatPass is disabled.";
    return Status::OK();
  }
  VLOG(1) << "TaoBaCEReformatPass is enabled.";

  for (auto *node : (*options.graph)->op_nodes()) {
    if (node->type_string() != "Const") {
      continue;
    }
    auto *dtype = node->attrs().Find("dtype");
    CHECK(dtype) << "Constant op has no `dtype` attr.";
    if (dtype->type() != DT_FLOAT) {
      continue;
    }
    auto *value = node->attrs().Find("value");
    CHECK(value) << "Constant op has no `value` attr.";
    auto tensor_proto = value->tensor();
    CHECK(tensor_proto.dtype() == dtype->type())
        << "Type of tensor proto in `value` attr match with `dtype` attr.";

    auto shape = tensor_proto.tensor_shape();
    if (shape.unknown_rank()) {
      VLOG(1) << "Ignore constant with unkown rank: " << node->name();
      continue;
    }
    if (shape.dim_size() != 2) {
      VLOG(1) << "Ignore non 2-dim constant: " << node->name();
      continue;
    }

    int64 max_dim = std::max(shape.dim(0).size(), shape.dim(1).size());
    int64 min_dim = std::min(shape.dim(0).size(), shape.dim(1).size());

    if ((max_dim > max_dim_bar_ && min_dim > min_dim_bar_) ||
        max_dim * min_dim > size_bar_) {
      VLOG(1) << "reformat constant:" << node->name();
      Tensor origin_const = Tensor();
      if (!origin_const.FromProto(tensor_proto)) {
        return errors::Internal(
            "TaoBaCEReformatPass parse constant value failed for node: ",
            node->name());
      }
      Tensor new_const = Tensor(DT_HALF, shape);
      CHECK(options.device_set != nullptr);
      CHECK(options.device_set->client_device() != nullptr);
      auto *device = options.device_set->client_device()->eigen_cpu_device();
      CHECK(device);

      new_const.flat<Eigen::half>().device(*device) =
          origin_const.flat<float>().cast<Eigen::half>();

      AttrValue new_value;
      auto *new_tensor_proto = new_value.mutable_tensor();
      new_tensor_proto->set_version_number(tensor_proto.version_number());
      if (!tensor_proto.tensor_content().empty()) {
        new_const.AsProtoTensorContent(new_tensor_proto);
      } else {
        new_const.AsProtoField(new_tensor_proto);
      }
      node->AddAttr("value", new_value);
      node->AddAttr("dtype", DT_HALF);
    }
  }
  return Status::OK();
}

} // namespace tao
} // namespace tensorflow

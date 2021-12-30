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

#include "mlir/mhlo/builder/batch_norm.h"

#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/mhlo/builder/mlir_attr_utils.h"

namespace mlir {
namespace mhlo {

mlir::Value BuildBatchNormInference(
    mlir::OpBuilder& builder, const mlir::Location& loc,
    const mlir::Value& input, const mlir::Value& scale,
    const mlir::Value& offset, const mlir::Value& mean,
    const mlir::Value& variance, float eps, mlir_dim_t feature_index) {
  mlir::FloatAttr eps_attr = builder.getF32FloatAttr(eps);
  mlir::IntegerAttr feature_index_attr =
      builder.getI64IntegerAttr(feature_index);

  auto bn_op = builder.create<mlir::mhlo::BatchNormInferenceOp>(
      loc, input.getType(), input, scale, offset, mean, variance, eps_attr,
      feature_index_attr);
  return bn_op.getResult();
}

}  // namespace mhlo
}  // namespace mlir

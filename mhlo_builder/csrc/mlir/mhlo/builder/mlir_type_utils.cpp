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

#include "mlir/mhlo/builder/mlir_type_utils.h"

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/mhlo/builder/macros.h"

namespace mlir {
namespace mhlo {

mlir::Type BuildMHloDimType(mlir::OpBuilder& builder) {
  return builder.getIntegerType(sizeof(mhlo_dim_t) * 8);
}

mlir::RankedTensorType GetMilrRankedTensorType(const mlir::Value& tensor) {
  auto ranked_value = tensor.getType().dyn_cast<mlir::RankedTensorType>();
  MHLO_CHECK(ranked_value, "the input tensor must be a RankedTensorType");
  return ranked_value;
}

mlir::Type GetMlirTensorElemType(const mlir::Value& value) {
  auto ranked_type = GetMilrRankedTensorType(value);
  return ranked_type.getElementType();
}

mlir::Value TryBuildElementTypeCast(mlir::OpBuilder& builder,
                                    const mlir::Location& loc,
                                    const mlir::Value& ml_tensor,
                                    const mlir::Type& elem_type) {
  auto ml_tensor_elem_type = GetMlirTensorElemType(ml_tensor);
  if (elem_type == ml_tensor_elem_type) {
    return ml_tensor;
  } else {
    return builder.create<mlir::mhlo::ConvertOp>(loc, ml_tensor, elem_type);
  }
}

mlir::RankedTensorType BuildRankedTensorTypeFrom(const mlir::Value& tensor) {
  auto inp_ranked_type = GetMilrRankedTensorType(tensor);
  std::vector<mlir_dim_t> ranked_shape(inp_ranked_type.getRank(),
                                       mlir::ShapedType::kDynamicSize);
  return mlir::RankedTensorType::get(ranked_shape,
                                     inp_ranked_type.getElementType());
}
}  // namespace mhlo
}  // namespace mlir

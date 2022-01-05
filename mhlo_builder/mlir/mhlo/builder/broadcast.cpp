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

#include "mlir/mhlo/builder/broadcast.h"

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/mhlo/builder/constant.h"
#include "mlir/mhlo/builder/mlir_attr_utils.h"
#include "mlir/mhlo/builder/mlir_shape_builder.h"
#include "mlir/mhlo/builder/mlir_type_utils.h"
#include "mlir/mhlo/builder/mlir_utils.h"

namespace mlir {
namespace mhlo {
mlir::Value BuildBroadcastScalarAsTensor(mlir::OpBuilder& builder,
                                         const mlir::Location& loc,
                                         const mlir::Value& scalar,
                                         const mlir::Value& tensor) {
  auto shape = BuildShapeOfTensor(builder, loc, tensor);
  auto broadcast = builder.create<mlir::mhlo::DynamicBroadcastInDimOp>(
      loc, tensor.getType(), scalar, shape, BuildI64ElementsAttr(builder, {}));
  return broadcast;
}

mlir::Value BuildBroadcastTensorInDims(
    mlir::OpBuilder& builder, const mlir::Location& loc,
    const mlir::Value& tensor, const SmallValueVec4& dims_size,
    const SmallVec4<mlir_dim_t>& broadcast_dims) {
  mlir_dim_t rank = dims_size.size();
  MHLO_CHECK(rank >= GetRankOfMlirValue(tensor),
             "can't broadcast from higher rank to lower");
  mlir::Value shape =
      builder.create<mlir::tensor::FromElementsOp>(loc, dims_size);

  SmallVec4<mlir_dim_t> output_shape =
      GetDimSizeListFromHloDimValList(dims_size);
  mlir::RankedTensorType result_type =
      mlir::RankedTensorType::get(output_shape, GetMlirTensorElemType(tensor));

  auto broadcast = builder.create<mlir::mhlo::DynamicBroadcastInDimOp>(
      loc, result_type, tensor, shape,
      BuildI64ElementsAttr(builder, broadcast_dims));
  return broadcast;
}

mlir::Value BuildBroadcastTensorAsOther(mlir::OpBuilder& builder,
                                        const mlir::Location& loc,
                                        const mlir::Value& tensor,
                                        const mlir::Value& other) {
  auto tensor_rank = GetRankOfMlirValue(tensor);
  auto higher_rank = GetRankOfMlirValue(other);
  MHLO_CHECK(higher_rank >= tensor_rank,
             "can't broadcast from higher rank to lower");
  if (tensor_rank == 0) {
    return BuildBroadcastScalarAsTensor(builder, loc, tensor, other);
  }
  auto leading_rank = higher_rank - tensor_rank;
  auto new_dim_sizes = BuildDimSizeListOfTensor(builder, loc, other);
  auto broadcast_dims = RangeIndices(leading_rank, higher_rank);
  return BuildBroadcastTensorInDims(builder, loc, tensor, new_dim_sizes,
                                    broadcast_dims);
}
}  // namespace mhlo
}  // namespace mlir

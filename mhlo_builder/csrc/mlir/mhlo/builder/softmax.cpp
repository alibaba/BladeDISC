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

#include "mlir/mhlo/builder/softmax.h"

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/mhlo/builder/broadcast.h"
#include "mlir/mhlo/builder/element_wise_binary.h"
#include "mlir/mhlo/builder/mlir_shape_builder.h"
#include "mlir/mhlo/builder/reduction.h"
#include "stablehlo/dialect/ChloOps.h"

namespace mlir {
namespace mhlo {

mlir::Value BuildSoftmax(mlir::OpBuilder& builder, const mlir::Location& loc,
                         const mlir::Value& ml_input, mlir_dim_t reduce_dim,
                         bool is_logsoftmax) {
  mlir_dim_t rank = GetRankOfMlirValue(ml_input);
  reduce_dim = NormalizeDimIndex(reduce_dim, rank);

  // do explicit broadcast on tensors
  SmallVec4<mlir_dim_t> broadcast_dims;
  for (mlir_dim_t k = 0; k < rank; ++k) {
    if (k == reduce_dim) continue;
    broadcast_dims.push_back(k);
  }
  auto input_shape = BuildDimSizeListOfTensor(builder, loc, ml_input);
  const auto& elem_type = GetMlirTensorElemType(ml_input);
  auto min_value = BuildHloMinValueForType(builder, loc, elem_type);
  const auto& max_val = BuildReduction<mlir::mhlo::MaxOp>(
      builder, loc, min_value, ml_input, {reduce_dim});
  const auto& broadcast_max = BuildBroadcastTensorInDims(
      builder, loc, max_val, input_shape, broadcast_dims);
  auto ml_input_centered = BuildMlirBinaryOp<mlir::chlo::BroadcastSubOp>(
      builder, loc, ml_input, broadcast_max, elem_type,
      /* no_implicit_broadcast */ true);
  const auto& exp_val =
      builder.create<mlir::mhlo::ExpOp>(loc, ml_input_centered).getResult();
  auto zero_value = BuildHloConstZeroForType(builder, loc, elem_type);
  const auto& sum_exp = BuildReduction<mlir::mhlo::AddOp>(
      builder, loc, zero_value, exp_val, {reduce_dim});
  if (is_logsoftmax) {
    const auto& log_sum_exp =
        builder.create<mlir::mhlo::LogOp>(loc, sum_exp).getResult();
    const auto& broadcast_log_sum = BuildBroadcastTensorInDims(
        builder, loc, log_sum_exp, input_shape, broadcast_dims);
    return BuildMlirBinaryOp<mlir::chlo::BroadcastSubOp>(
        builder, loc, ml_input_centered, broadcast_log_sum, elem_type,
        /* no_implicit_broadcast */ true);
  } else {
    const auto& broadcast_sum = BuildBroadcastTensorInDims(
        builder, loc, sum_exp, input_shape, broadcast_dims);
    return BuildMlirBinaryOp<mlir::chlo::BroadcastDivOp>(
        builder, loc, exp_val, broadcast_sum, elem_type,
        /* no_implicit_broadcast */ true);
  }
}

}  // namespace mhlo
}  // namespace mlir

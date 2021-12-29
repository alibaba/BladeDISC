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

#include "mlir/mhlo/builder/matmul.h"

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/mhlo/builder/broadcast.h"
#include "mlir/mhlo/builder/mlir_attr_utils.h"
#include "mlir/mhlo/builder/mlir_shape_builder.h"
#include "mlir/mhlo/builder/mlir_utils.h"

namespace mlir {
namespace mhlo {

mlir::Value BuildDotProduct(mlir::OpBuilder& builder, const mlir::Location& loc,
                            const mlir::Value& lhs, const mlir::Value& rhs,
                            mlir_dim_t rank) {
  MHLO_CHECK(rank >= 2, "The input of DotProduct must has rank >= 2");
  auto lhs_contracting_dims_attr =
      BuildI64ElementsAttr(builder, SmallVec4<mlir_dim_t>({rank - 1}));
  auto rhs_contracting_dims_attr =
      BuildI64ElementsAttr(builder, SmallVec4<mlir_dim_t>({rank - 2}));
  SmallVec4<mlir_dim_t> batch_dims;
  for (mlir_dim_t r = 0; r < rank - 2; ++r) {
    batch_dims.push_back(r);
  }
  // lhs_shape[b, m, n], rhs_shape[b', n', k] -> result_shape[b, m, k],
  // assert b == b' and n == n', but we could only verify it at runtime
  auto lhs_shape = GetMilrRankedTensorType(lhs).getShape();
  auto rhs_shape = GetMilrRankedTensorType(rhs).getShape();
  SmallVec4<mlir_dim_t> result_shape(lhs_shape.begin(), lhs_shape.end());
  result_shape[rank - 1] = rhs_shape[rank - 1];

  auto elem_type = GetMlirTensorElemType(lhs);
  auto result_type = mlir::RankedTensorType::get(result_shape, elem_type);

  auto lhs_batch_dims_attr = BuildI64ElementsAttr(builder, batch_dims);
  auto rhs_batch_dims_attr = BuildI64ElementsAttr(builder, batch_dims);
  auto dot_dimension_attr = mlir::mhlo::DotDimensionNumbers::get(
      lhs_batch_dims_attr, rhs_batch_dims_attr, lhs_contracting_dims_attr,
      rhs_contracting_dims_attr, builder.getContext());
  auto result = builder.create<mlir::mhlo::DotGeneralOp>(
      loc, result_type, lhs, rhs, dot_dimension_attr,
      /*precision_config*/ nullptr);
  return result.getResult();
}

mlir::Value BuildDotProduct_bmm(mlir::OpBuilder& builder,
                                const mlir::Location& loc,
                                const mlir::Value& inp_lhs,
                                const mlir::Value& inp_rhs) {
  mlir::Value lhs = inp_lhs;
  mlir::Value rhs = inp_rhs;

  auto lhs_rank = GetRankOfMlirValue(lhs);
  auto rhs_rank = GetRankOfMlirValue(rhs);
  MHLO_CHECK(lhs_rank >= 2, "The input of batch-matmul must has rank >= 2");
  MHLO_CHECK(rhs_rank >= 2, "The input of batch-matmul must has rank >= 2");
  // The non-matrix (i.e. batch) dimensions are broadcasted (and thus must be
  // broadcastable).
  auto max_rank = std::max(lhs_rank, rhs_rank);
  auto min_rank = std::min(lhs_rank, rhs_rank);
  if (max_rank != min_rank) {
    auto leading_rank = max_rank - min_rank;
    auto leading_dims = RangeIndices(0, leading_rank);
    auto broadcast_dims = RangeIndices(leading_rank, max_rank);
    if (lhs_rank < rhs_rank) {
      auto new_dim_sizes =
          BuildDimSizeListOfTensor(builder, loc, rhs, leading_dims);
      auto lhs_dim_sizes = BuildDimSizeListOfTensor(builder, loc, lhs);
      new_dim_sizes.append(lhs_dim_sizes.begin(), lhs_dim_sizes.end());
      lhs = BuildBroadcastTensorInDims(builder, loc, lhs, new_dim_sizes,
                                       broadcast_dims);
    } else {
      auto new_dim_sizes =
          BuildDimSizeListOfTensor(builder, loc, lhs, leading_dims);
      auto rhs_dim_sizes = BuildDimSizeListOfTensor(builder, loc, rhs);
      new_dim_sizes.append(rhs_dim_sizes.begin(), rhs_dim_sizes.end());
      rhs = BuildBroadcastTensorInDims(builder, loc, rhs, new_dim_sizes,
                                       broadcast_dims);
    }
  }

  // Maybe a bug
  // [?, ?, m, n] x [?, n, k] ==> batch_matmul([m,n], [n,k])
  return BuildDotProduct(builder, loc, lhs, rhs, /*rank*/ max_rank);
}

mlir::Value BuildDotProduct_mm(mlir::OpBuilder& builder,
                               const mlir::Location& loc,
                               const mlir::Value& inp_lhs,
                               const mlir::Value& inp_rhs) {
  mlir::Value lhs = inp_lhs;
  mlir::Value rhs = inp_rhs;

  auto rhs_rank = GetRankOfMlirValue(rhs);
  auto lhs_rank = GetRankOfMlirValue(lhs);
  MHLO_CHECK(rhs_rank == 2,
             "The right hand-side input of matmul must has rank == 2");
  MHLO_CHECK(lhs_rank >= 2,
             "The left hand-side input of matmul must has rank >= 2");

  // [?, m, n] x [n, k] ==> [?xm, n] x [n, k]
  SmallValueVec4 claps_dim_values;
  if (lhs_rank > 2) {
    SmallVec4<mlir_dim_t> clap_dims;
    for (size_t d = 0; d < lhs_rank - 1; ++d) {
      clap_dims.push_back(d);
    }
    std::tie(lhs, claps_dim_values) =
        BuildCollapseTensorShape(builder, loc, lhs, clap_dims);
  }
  auto result = BuildDotProduct(builder, loc, lhs, rhs, /*rank*/ 2);
  return BuildExpandTensorShapeWithDhloDims(builder, loc, result,
                                            claps_dim_values, /*expand_pos*/ 0);
}

mlir::Value BuildDotProduct_mv(mlir::OpBuilder& builder,
                               const mlir::Location& loc,
                               const mlir::Value& inp_lhs,
                               const mlir::Value& inp_rhs) {
  mlir::Value lhs = inp_lhs;
  mlir::Value rhs = inp_rhs;

  auto rhs_rank = GetRankOfMlirValue(rhs);
  auto lhs_rank = GetRankOfMlirValue(lhs);
  MHLO_CHECK(rhs_rank == 1,
             "The right hand-side input of matmul must has rank == 1");
  MHLO_CHECK(lhs_rank >= 2,
             "The left hand-side input of matmul must has rank >= 2");

  rhs = BuildUnsqueezeTensorShape(builder, loc, rhs, {1});
  auto output = BuildDotProduct_mm(builder, loc, lhs, rhs);
  mlir::Value product;
  SmallValueVec4 claps_dim_values;
  std::tie(product, claps_dim_values) =
      BuildCollapseTensorShape(builder, loc, output, {-2, -1});
  return product;
}
}  // namespace mhlo
}  // namespace mlir

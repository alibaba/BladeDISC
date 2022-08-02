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

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/mhlo/builder/mlir_shape_builder.h"
#include "mlir/mhlo/builder/mlir_utils.h"
#include "mlir/mhlo/builder/standard.h"

namespace mlir {
namespace mhlo {

mlir::Value BuildStdNormalizeIndex(mlir::OpBuilder& builder,
                                   const mlir::Location& loc,
                                   const mlir::Value& index,
                                   const mlir::Value& dim_size,
                                   const mlir::Value& neg_dim_size) {
  // index_bounded = min(max(-dim_size, index), dim_size)
  auto index_bounded = BuildStdMaximumSigned(builder, loc, neg_dim_size, index);
  index_bounded = BuildStdMinimumSigned(builder, loc, dim_size, index_bounded);
  // By avoiding x rem 0 in case dim_size is zero. The logic should be:
  //   if dim_size == 0:
  //       result = 0;
  //   else:
  //       result = index rem dim_size
  // However, we don't want an "if" here so we simply guard dim_size from being
  // zero
  auto zero = BuildStdConstLike(builder, loc, 0, index);
  auto one = BuildStdConstLike(builder, loc, 1, index);
  auto not_zero = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::ne, dim_size, zero);
  auto dim_size_no_zero =
      builder.create<mlir::arith::SelectOp>(loc, not_zero, dim_size, one);
  // remainder = (dim_size + index_bounded) % dim_size
  auto remainder = BuildStdRemainderSigned(
      builder, loc, BuildStdAddSigned(builder, loc, dim_size, index_bounded),
      dim_size_no_zero);

  // cond = index >= dim_size
  auto cond = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::sge, index, dim_size);
  // cond ? dim_size: remainder
  return builder.create<mlir::arith::SelectOp>(loc, cond, dim_size, remainder);
}

mlir::Value BuildDynamicSliceInternal(mlir::OpBuilder& builder,
                                      const mlir::Location& loc,
                                      const mlir::Value& input,
                                      const mlir::Value& start_index,
                                      const mlir::Value& end_index,
                                      const mlir::Value& step,
                                      mlir_dim_t dim_index) {
  // start_index & end_index has been normailized into range [0, dim_size]
  auto dim_size = builder.create<tensor::DimOp>(loc, input, dim_index);
  // cast i64 -> index
  auto norm_end_index = BuildStdScalarToIndexType(builder, loc, end_index);
  auto norm_start_index = BuildStdScalarToIndexType(builder, loc, start_index);
  auto norm_step_index = BuildStdScalarToIndexType(builder, loc, step);

  auto mhlo_zero = BuildStdConstForI32(builder, loc, 0);
  auto mhlo_one = BuildStdConstForI32(builder, loc, 1);
  auto mhlo_dim_type = BuildMHloDimType(builder);
  auto mhlo_dim_size =
      builder.create<mlir::arith::IndexCastOp>(loc, mhlo_dim_type, dim_size);

  SmallValueVec4 start_indices;
  SmallValueVec4 end_indices;
  SmallValueVec4 strides;
  mlir_dim_t rank = GetRankOfMlirValue(input);
  start_indices.reserve(rank);
  end_indices.reserve(rank);
  strides.reserve(rank);
  for (mlir_dim_t r = 0; r < rank; ++r) {
    if (r == dim_index) {
      auto mhlo_start_index = builder.create<mlir::arith::IndexCastOp>(
          loc, mhlo_dim_type, norm_start_index);
      start_indices.push_back(mhlo_start_index);
      auto mhlo_end_index = builder.create<mlir::arith::IndexCastOp>(
          loc, mhlo_dim_type, norm_end_index);
      end_indices.push_back(mhlo_end_index);
      auto mhlo_step_index = builder.create<mlir::arith::IndexCastOp>(
          loc, mhlo_dim_type, norm_step_index);
      strides.push_back(mhlo_step_index);
    } else {
      start_indices.push_back(mhlo_zero);
      end_indices.push_back(BuildHloDimSizeOfTensor(builder, loc, input, r));
      strides.push_back(mhlo_one);
    }
  }

  auto start_tensor = BuildFromElements(builder, loc, start_indices);
  auto end_tensor = BuildFromElements(builder, loc, end_indices);
  auto strides_tensor = BuildFromElements(builder, loc, strides);

  auto ranked_type = GetMilrRankedTensorType(input);
  auto input_shape = ranked_type.getShape();
  SmallVec4<mlir_dim_t> slice_shape(input_shape.begin(), input_shape.end());
  slice_shape[dim_index] = mlir::ShapedType::kDynamicSize;
  auto slice_output_type =
      mlir::RankedTensorType::get(slice_shape, ranked_type.getElementType());
  return builder.create<mlir::mhlo::RealDynamicSliceOp>(
      loc, slice_output_type, input, start_tensor, end_tensor, strides_tensor);
}

mlir::Value BuildDynamicSlice(mlir::OpBuilder& builder,
                              const mlir::Location& loc,
                              const mlir::Value& input,
                              const mlir::Value& start_index,
                              const mlir::Value& end_index,
                              const mlir::Value& step, mlir_dim_t dim_index) {
  auto rank = GetRankOfMlirValue(input);
  auto norm_dim_idx = NormalizeDimIndex(dim_index, rank);
  auto dim_size = BuildStdDimSizeOfTensor(builder, loc, input, norm_dim_idx);
  auto neg_dim_size = BuildStdNegtive(builder, loc, dim_size);
  auto norm_start_index =
      BuildStdNormalizeIndex(builder, loc, start_index, dim_size, neg_dim_size);
  auto norm_end_index =
      BuildStdNormalizeIndex(builder, loc, end_index, dim_size, neg_dim_size);
  return BuildDynamicSliceInternal(builder, loc, input, norm_start_index,
                                   norm_end_index, step, norm_dim_idx);
}

std::tuple<mlir::Value, mlir::Value> BuildHalfSplit(mlir::OpBuilder& builder,
                                                    const mlir::Location& loc,
                                                    const mlir::Value& input,
                                                    mlir_dim_t dim_index) {
  auto rank = GetRankOfMlirValue(input);
  dim_index = NormalizeDimIndex(dim_index, rank);
  auto dim_size = BuildStdDimSizeOfTensor(builder, loc, input, dim_index);
  auto std_zero = BuildStdConstForI64(builder, loc, 0);
  auto std_one = BuildStdConstForI64(builder, loc, 1);
  auto std_two = BuildStdConstForI64(builder, loc, 2);
  auto half_dim_size = BuildStdDivSigned(builder, loc, dim_size, std_two);
  auto lhs = BuildDynamicSliceInternal(builder, loc, input, std_zero,
                                       half_dim_size, std_one, dim_index);
  auto rhs = BuildDynamicSliceInternal(builder, loc, input, half_dim_size,
                                       dim_size, std_one, dim_index);
  return std::make_tuple(lhs, rhs);
}

mlir::Value BuildSelect(mlir::OpBuilder& builder, const mlir::Location& loc,
                        const mlir::Value& input,
                        const mlir::Value& select_index, mlir_dim_t dim_index) {
  auto rank = GetRankOfMlirValue(input);
  dim_index = NormalizeDimIndex(dim_index, rank);
  auto dim_size = BuildStdDimSizeOfTensor(builder, loc, input, dim_index);
  auto neg_dim_size = BuildStdNegtive(builder, loc, dim_size);
  auto std_one = BuildStdConstForI64(builder, loc, 1);
  auto start_index = BuildStdNormalizeIndex(builder, loc, select_index,
                                            dim_size, neg_dim_size);
  auto end_index = BuildStdAddSigned(builder, loc, start_index, std_one);
  auto result = BuildDynamicSliceInternal(builder, loc, input, start_index,
                                          end_index, std_one, dim_index);
  SmallValueVec4 new_dim_sizes;
  new_dim_sizes.reserve(rank - 1);
  for (mlir_dim_t k = 0; k < rank; ++k) {
    if (k != dim_index) {
      new_dim_sizes.push_back(BuildHloDimSizeOfTensor(builder, loc, result, k));
    }
  }
  return BuildDynamicReshapeTensor(builder, loc, result, new_dim_sizes);
}

mlir::Value BuildRoll(mlir::OpBuilder& builder, const mlir::Location& loc,
                      const mlir::Value& input, mlir_dim_t shift,
                      mlir_dim_t dim) {
  // roll(input, shift, dim) = cat(
  //   slice(input, (dim_size - shift) % dim_size, dim_size),
  //   slice(input, 0, - shift))
  auto dim_size = BuildStdDimSizeOfTensor(builder, loc, input, dim);
  auto std_zero = BuildStdConstForI64(builder, loc, 0);
  auto std_one = BuildStdConstForI64(builder, loc, 1);
  auto std_shift = BuildStdConstForI64(builder, loc, shift);
  auto split_pos = BuildStdSubSigned(builder, loc, dim_size, std_shift);
  split_pos = BuildStdRemainderSigned(builder, loc, split_pos, dim_size);

  auto lhs = BuildDynamicSliceInternal(builder, loc, input, split_pos, dim_size,
                                       std_one, dim);
  auto rhs = BuildDynamicSliceInternal(builder, loc, input, std_zero, split_pos,
                                       std_one, dim);

  auto ranked_type = GetMilrRankedTensorType(input);
  auto result = builder.create<mlir::mhlo::ConcatenateOp>(
      loc, ranked_type, SmallVec4<mlir::Value>{lhs, rhs},
      builder.getI64IntegerAttr(dim));
  return result;
}

}  // namespace mhlo
}  // namespace mlir

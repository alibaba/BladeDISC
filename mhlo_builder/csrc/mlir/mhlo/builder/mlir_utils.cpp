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

#include "mlir/mhlo/builder/mlir_utils.h"

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/mhlo/builder/constant.h"
#include "mlir/mhlo/builder/standard.h"

namespace mlir {
namespace mhlo {
namespace {
template <class MLIR_T>
std::string toString(MLIR_T value) {
  std::string s;
  llvm::raw_string_ostream ss(s);
  ss << value;
  return ss.str();
}
}  // namespace

SmallVec4<mlir_dim_t> RangeIndices(mlir_dim_t min, mlir_dim_t max) {
  SmallVec4<mlir_dim_t> range;
  for (mlir_dim_t k = min; k < max; ++k) {
    range.push_back(k);
  }
  return range;
}

mlir_dim_t GetRankOfMlirValue(const mlir::Value& tensor) {
  auto ranked_value = tensor.getType().dyn_cast<mlir::RankedTensorType>();
  MHLO_CHECK(ranked_value, "the input tensor must be a RankedTensorType");
  auto rank = ranked_value.getRank();
  return rank;
}

llvm::Optional<mlir_dim_t> GetLenIfStaticLenVector(const mlir::Value& tensor) {
  RankedTensorType ranked_type = GetMilrRankedTensorType(tensor);
  const auto& shape = ranked_type.getShape();
  if (shape.size() == 1 && shape[0] >= 0) {
    return shape[0];
  } else {
    return llvm::None;
  }
}

mlir_dim_t NormalizeDimIndex(mlir_dim_t dim_index, mlir_dim_t rank) {
  MHLO_CHECK(dim_index >= -rank && dim_index < rank,
             "the dimension must be in range [", -rank, ", ", rank, ")");
  return (dim_index + rank) % rank;
}

SmallVec4<mlir_dim_t> NormalizeDimIndex(const SmallVec4<mlir_dim_t>& dims,
                                        mlir_dim_t rank) {
  SmallVec4<mlir_dim_t> new_dims;
  std::transform(dims.begin(), dims.end(), std::back_inserter(new_dims),
                 [rank](mlir_dim_t d) -> mlir_dim_t {
                   return NormalizeDimIndex(d, rank);
                 });
  return new_dims;
}

mlir::Value BuildStandardI32NumelOfTensor(
    mlir::OpBuilder& builder, const mlir::Location& loc,
    const mlir::Value& input, const SmallVec4<mlir_dim_t>& numel_dims) {
  mlir::Value num_elem = BuildStdConstForI32(builder, loc, 1);
  mlir::Type mhlo_dim_type = BuildMHloDimType(builder);
  for (mlir_dim_t dim : numel_dims) {
    mlir::Value dim_size = builder.create<tensor::DimOp>(loc, input, dim);
    dim_size =
        builder.create<mlir::arith::IndexCastOp>(loc, mhlo_dim_type, dim_size);
    num_elem = builder.create<mlir::arith::MulIOp>(loc, num_elem, dim_size);
  }
  return num_elem;
}

mlir::Value BuildHloNumelOfTensor(mlir::OpBuilder& builder,
                                  const mlir::Location& loc,
                                  const mlir::Value& input,
                                  const SmallVec4<mlir_dim_t>& numel_dims) {
  auto num_elem =
      BuildStandardI32NumelOfTensor(builder, loc, input, numel_dims);
  // TODO: num_elem_tensor must be an rank 0 tensor, but
  // mlir::tensor::FromElementsOp doesn't support creation of rank
  // 0 tensor. To workaround, we first make an rank 1 tensor, then reshape to
  // rank 0.
  mlir::Type mhlo_dim_type = BuildMHloDimType(builder);
  mlir::Value num_elem_tensor = builder.create<mlir::tensor::FromElementsOp>(
      loc, mlir::ArrayRef<mlir::Value>({num_elem}));
  num_elem_tensor = builder.create<mlir::mhlo::ReshapeOp>(
      loc,
      mlir::RankedTensorType::get(mlir::ArrayRef<mlir_dim_t>{}, mhlo_dim_type),
      num_elem_tensor);
  return num_elem_tensor;
}

// Returns minimal value for the given int or float element type.
mlir::Value BuildHloMinValueForType(mlir::OpBuilder& builder,
                                    const mlir::Location& loc,
                                    const mlir::Type& elem_type) {
  mlir::RankedTensorType scalar_ty = mlir::RankedTensorType::get({}, elem_type);
  mlir::DenseElementsAttr attr;
  if (auto float_ty = elem_type.dyn_cast_or_null<mlir::FloatType>()) {
    llvm::APFloat neg_inf =
        llvm::APFloat::getInf(float_ty.getFloatSemantics(), /*negative=*/true);
    attr = mlir::DenseElementsAttr::get(scalar_ty, neg_inf);
  } else if (auto int_ty = elem_type.dyn_cast_or_null<mlir::IntegerType>()) {
    if (int_ty.isSigned()) {
      // Gets minimum signed value of APInt for a specific bit width.
      llvm::APInt min_val = llvm::APInt::getSignedMinValue(int_ty.getWidth());
      attr = mlir::DenseElementsAttr::get(scalar_ty, min_val);
    } else {
      // Gets minimum unsigned value of APInt for a specific bit width.
      llvm::APInt min_val = llvm::APInt::getMinValue(int_ty.getWidth());
      attr = mlir::DenseElementsAttr::get(scalar_ty, min_val);
    }
  } else {
    MHLO_CHECK(false, toString(elem_type), " unsupported");
  }
  return builder.create<mlir::mhlo::ConstantOp>(loc, attr);
}

// Returns maximal value for the given int or float element type.
mlir::Value BuildHloMaxValueForType(mlir::OpBuilder& builder,
                                    const mlir::Location& loc,
                                    const mlir::Type& elem_type) {
  mlir::RankedTensorType scalar_ty = mlir::RankedTensorType::get({}, elem_type);
  mlir::DenseElementsAttr attr;
  if (auto float_ty = elem_type.dyn_cast_or_null<mlir::FloatType>()) {
    llvm::APFloat pos_inf =
        llvm::APFloat::getInf(float_ty.getFloatSemantics(), /*negative=*/false);
    attr = mlir::DenseElementsAttr::get(scalar_ty, pos_inf);
  } else if (auto int_ty = elem_type.dyn_cast_or_null<mlir::IntegerType>()) {
    if (int_ty.isSigned()) {
      // Gets maximum signed value of APInt for a specific bit width.
      llvm::APInt min_val = llvm::APInt::getSignedMaxValue(int_ty.getWidth());
      attr = mlir::DenseElementsAttr::get(scalar_ty, min_val);
    } else {
      // Gets maximum unsigned value of APInt for a specific bit width.
      llvm::APInt min_val = llvm::APInt::getMaxValue(int_ty.getWidth());
      attr = mlir::DenseElementsAttr::get(scalar_ty, min_val);
    }
  } else {
    MHLO_CHECK(false, toString(elem_type), " unsupported");
  }
  return builder.create<mlir::mhlo::ConstantOp>(loc, attr);
}

// Returns int or float DenseElementsAttr of const 0
mlir::Value BuildHloConstOneForType(mlir::OpBuilder& builder,
                                    const mlir::Location& loc,
                                    const mlir::Type& elem_type) {
  mlir::RankedTensorType scalar_ty = mlir::RankedTensorType::get({}, elem_type);
  mlir::DenseElementsAttr const_attr;
  if (auto float_ty = elem_type.dyn_cast_or_null<mlir::FloatType>()) {
    mlir::FloatAttr attr = mlir::FloatAttr::get(float_ty, 1);
    const_attr = mlir::DenseElementsAttr::get(scalar_ty, attr);
  } else if (auto int_ty = elem_type.dyn_cast_or_null<mlir::IntegerType>()) {
    mlir::IntegerAttr attr = mlir::IntegerAttr::get(int_ty, 1);
    const_attr = mlir::DenseElementsAttr::get(scalar_ty, attr);
  } else {
    MHLO_CHECK(false, "only mlir::FloatType and mlir::IntType are supported");
  }

  return builder.create<mlir::mhlo::ConstantOp>(loc, const_attr);
}

// Returns int or float DenseElementsAttr of const 0
mlir::Value BuildHloConstZeroForType(mlir::OpBuilder& builder,
                                     const mlir::Location& loc,
                                     const mlir::Type& elem_type) {
  mlir::RankedTensorType scalar_ty = mlir::RankedTensorType::get({}, elem_type);
  mlir::DenseElementsAttr const_attr;
  if (auto float_ty = elem_type.dyn_cast_or_null<mlir::FloatType>()) {
    mlir::FloatAttr attr = mlir::FloatAttr::get(float_ty, 0);
    const_attr = mlir::DenseElementsAttr::get(scalar_ty, attr);
  } else if (auto int_ty = elem_type.dyn_cast_or_null<mlir::IntegerType>()) {
    mlir::IntegerAttr attr = mlir::IntegerAttr::get(int_ty, 0);
    const_attr = mlir::DenseElementsAttr::get(scalar_ty, attr);
  } else {
    MHLO_CHECK(false, "only mlir::FloatType and mlir::IntType are supported");
  }

  return builder.create<mlir::mhlo::ConstantOp>(loc, const_attr);
}

mlir::Value BuildExtractVectorElement(mlir::OpBuilder& builder,
                                      const mlir::Location& loc,
                                      const mlir::Value& vector_value,
                                      mlir_dim_t index) {
  MHLO_CHECK(1 == GetRankOfMlirValue(vector_value),
             "the input value should have rank 1");
  auto idx_value = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getIntegerAttr(builder.getIndexType(), index));
  return builder.create<tensor::ExtractOp>(loc, vector_value,
                                           mlir::ValueRange{idx_value});
}

// Returns float DenseElementsAttr of const value
mlir::Value BuildHloConstForFloatType(mlir::OpBuilder& builder,
                                      const mlir::Location& loc,
                                      const mlir::Type& elem_type,
                                      const float value) {
  mlir::RankedTensorType scalar_ty = mlir::RankedTensorType::get({}, elem_type);
  mlir::DenseElementsAttr const_attr;
  if (auto float_ty = elem_type.dyn_cast_or_null<mlir::FloatType>()) {
    mlir::FloatAttr attr = mlir::FloatAttr::get(float_ty, value);
    const_attr = mlir::DenseElementsAttr::get(scalar_ty, attr);
  } else {
    MHLO_CHECK(false, "only mlir::FloatType is supported");
  }
  return builder.create<mlir::mhlo::ConstantOp>(loc, const_attr);
}

mlir::Value BuildResolveUnknownDimSizeI32(mlir::OpBuilder& builder,
                                          const mlir::Location& loc,
                                          const mlir::Value& input,
                                          const SmallValueVec4& i32_dim_sizes) {
  // NB: This builder function work fine when there are no more than one
  // negative value in dim sizes, and the negative value must be -1. Otherwise,
  // the behavior is undefined at compile time.
  //
  // We could not catch error at compile time, because the dim size value maybe
  // generated at runtime.

  mlir_dim_t input_rank = GetRankOfMlirValue(input);
  // number of element of input tensor
  const auto& input_numel = BuildStandardI32NumelOfTensor(
      builder, loc, input, RangeIndices(0, input_rank));
  SmallValueVec4 shape_elements;

  // minus reduce product of shape elements
  // for example,
  // given shape [b, -1, w, h]:
  //   minus reduce_prod(shape) = -1 * b * -1 * w * h = b * w * h
  // given shape [b, c, w, h]:
  //   minus reduce_prod(shape) = -1 * b * c * w * h
  auto minus_1 = BuildStdConstForI32(builder, loc, -1);
  auto reduce_prod = minus_1;
  mlir_dim_t output_rank = i32_dim_sizes.size();
  for (mlir_dim_t k = 0; k < output_rank; ++k) {
    auto dim_size = i32_dim_sizes[k];
    reduce_prod =
        builder.create<mlir::arith::MulIOp>(loc, dim_size, reduce_prod);
    shape_elements.push_back(dim_size);
  }

  for (mlir_dim_t k = 0; k < output_rank; ++k) {
    auto dim_size = shape_elements[k];
    auto is_minus_1 = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, dim_size, minus_1);
    // is_minus_1? (input_numel/reduce_prod) : dim_size
    auto resolved_dim_size = builder.create<mlir::arith::SelectOp>(
        loc, is_minus_1,
        builder.create<mlir::arith::DivSIOp>(loc, input_numel, reduce_prod),
        dim_size);
    shape_elements[k] = resolved_dim_size;
  }

  return builder.create<mlir::tensor::FromElementsOp>(loc, shape_elements);
}

mlir::Value BuildReshapeTensorToScalar(mlir::OpBuilder& builder,
                                       const mlir::Location& loc,
                                       const mlir::Value& single_value_input) {
  auto ranked_type = GetMilrRankedTensorType(single_value_input);
  auto shape = ranked_type.getShape();
  MHLO_CHECK(shape.size() == 1 && shape[0] == 1,
             "Could not reshape non-single value tensor to scalar");
  return builder.create<mlir::mhlo::ReshapeOp>(
      loc,
      mlir::RankedTensorType::get(mlir::ArrayRef<mlir_dim_t>{},
                                  ranked_type.getElementType()),
      single_value_input);
}

SmallVec4<mlir_dim_t> GetDimSizeListFromHloDimValList(
    const SmallValueVec4& dim_vals) {
  SmallVec4<mlir_dim_t> output_shape;
  output_shape.reserve(dim_vals.size());
  for (auto dim_val : dim_vals) {
    if (auto dim_size = CastStdConstToI64(dim_val)) {
      output_shape.push_back(*dim_size);
    } else {
      output_shape.push_back(mlir::ShapedType::kDynamicSize);
    }
  }
  return output_shape;
}

}  // namespace mhlo
}  // namespace mlir

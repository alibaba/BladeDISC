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

#include "mlir/mhlo/builder/mlir_shape_builder.h"

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/mhlo/builder/constant.h"
#include "mlir/mhlo/builder/mlir_attr_utils.h"
#include "mlir/mhlo/builder/mlir_type_utils.h"
#include "mlir/mhlo/builder/mlir_utils.h"
#include "mlir/mhlo/builder/standard.h"

namespace mlir {
namespace mhlo {

mlir::Value BuildHloDimSizeOfTensor(mlir::OpBuilder& builder,
                                    const mlir::Location& loc,
                                    const mlir::Value& tensor,
                                    mlir_dim_t dim_index) {
  auto ranked_type = GetMilrRankedTensorType(tensor);
  auto rank = ranked_type.getRank();
  dim_index = NormalizeDimIndex(dim_index, rank);
  auto dim_size = ranked_type.getDimSize(dim_index);
  if (dim_size == mlir::ShapedType::kDynamicSize) {
    return builder.create<mlir::arith::IndexCastOp>(
        loc, BuildMHloDimType(builder),
        builder.create<tensor::DimOp>(loc, tensor, dim_index));
  } else {
    return BuildStdConstForI32(builder, loc, dim_size);
  }
}

SmallValueVec4 BuildDimSizeListOfTensor(mlir::OpBuilder& builder,
                                        const mlir::Location& loc,
                                        const mlir::Value& tensor,
                                        const SmallVec4<mlir_dim_t>& dims) {
  auto rank = GetRankOfMlirValue(tensor);
  auto norm_dims =
      dims.size() ? NormalizeDimIndex(dims, rank) : RangeIndices(0, rank);

  SmallValueVec4 shape_values;
  shape_values.reserve(rank);

  for (mlir_dim_t d : norm_dims) {
    shape_values.push_back(BuildHloDimSizeOfTensor(builder, loc, tensor, d));
  }
  return shape_values;
}

mlir::Value BuildShapeOfTensor(mlir::OpBuilder& builder,
                               const mlir::Location& loc,
                               const mlir::Value& tensor) {
  auto shape_values = BuildDimSizeListOfTensor(builder, loc, tensor);
  int rank = GetRankOfMlirValue(tensor);

  mlir::Value shape =
      builder.create<mlir::tensor::FromElementsOp>(loc, shape_values);
  return shape;
}

mlir::Value BuildDynamicReshapeTensor(mlir::OpBuilder& builder,
                                      const mlir::Location& loc,
                                      const mlir::Value& tensor,
                                      const SmallValueVec4& new_shape_vals) {
  // create mhlo::DynamicReshapeOp
  int new_rank = new_shape_vals.size();
  mlir::Value shape =
      builder.create<mlir::tensor::FromElementsOp>(loc, new_shape_vals);

  SmallVec4<mlir_dim_t> output_shape =
      GetDimSizeListFromHloDimValList(new_shape_vals);
  mlir::RankedTensorType result_type =
      mlir::RankedTensorType::get(output_shape, GetMlirTensorElemType(tensor));
  return builder.create<mlir::mhlo::DynamicReshapeOp>(loc, result_type, tensor,
                                                      shape);
}

mlir::Value BuildUnsqueezeTensorShape(
    mlir::OpBuilder& builder, const mlir::Location& loc,
    const mlir::Value& tensor, const SmallVec4<mlir_dim_t>& input_unsqz_dims) {
  // Returns a new tensor with dims of size 1 inserted at the specified
  // position.
  //
  // The position indices (must be high to low dimension number of the returned
  // tensor) are specified with unsqz_dims. Indices must be in-order, and in
  // range of tensor rank. Thus, unsqueeze a rank 1 tensor with {0, 2}, {0, 1,
  // 3}, {0, 1, 2} are all valid dimension sets, but {0, 3}, {2} are not.
  auto rank = GetRankOfMlirValue(tensor);
  size_t new_rank = rank + input_unsqz_dims.size();
  auto unsqz_dims = NormalizeDimIndex(input_unsqz_dims, new_rank);
  for (size_t k = 0; k < unsqz_dims.size(); ++k) {
    if (k > 1) {
      MHLO_CHECK(unsqz_dims[k] > unsqz_dims[k - 1],
                 "unsqueeze dimensions must be specified in order.")
    }
  }

  // get original tensor shape in mlir standard dialect
  auto shape_values = BuildDimSizeListOfTensor(builder, loc, tensor);

  SmallValueVec4 new_shape_vals;
  auto std_i32 = BuildStdConstForI32(builder, loc, 1);
  for (size_t k = 0, i = 0, j = 0; k < new_rank; ++k) {
    if (j < unsqz_dims.size() && unsqz_dims[j] == k) {
      new_shape_vals.push_back(std_i32);
      j++;
    } else {
      new_shape_vals.push_back(shape_values[i++]);
    }
  }
  return BuildDynamicReshapeTensor(builder, loc, tensor, new_shape_vals);
}

mlir::Value BuildSqueezeTensorShape(mlir::OpBuilder& builder,
                                    const mlir::Location& loc,
                                    const mlir::Value& tensor,
                                    const SmallVec4<mlir_dim_t>& sqz_dims) {
  auto tensor_ranked_type = GetMilrRankedTensorType(tensor);
  int rank = tensor_ranked_type.getRank();
  auto norm_sqz_dims = NormalizeDimIndex(sqz_dims, rank);
  for (size_t k = 0; k < norm_sqz_dims.size(); ++k) {
    if (k > 1) {
      MHLO_CHECK(norm_sqz_dims[k] > norm_sqz_dims[k - 1],
                 "squeeze dimensions must be specified in order.")
    }
  }

  SmallValueVec4 new_shape_vals;
  new_shape_vals.reserve(rank);
  // The squeezed dim_sizes are considered to be 1,
  // otherwise the compilation behaviors are undefined.
  for (int r = 0, j = 0; r < rank; ++r) {
    if (j < norm_sqz_dims.size() && norm_sqz_dims[j] == r) {
      // skip the dimension to squeeze
      j++;
      continue;
    } else {
      new_shape_vals.push_back(
          BuildHloDimSizeOfTensor(builder, loc, tensor, r));
    }
  }

  return BuildDynamicReshapeTensor(builder, loc, tensor, new_shape_vals);
}

std::tuple<mlir::Value, SmallValueVec4> BuildCollapseTensorShape(
    mlir::OpBuilder& builder, const mlir::Location& loc,
    const mlir::Value& tensor, const SmallVec4<mlir_dim_t>& inp_clap_dims) {
  // Ref to XLA:Collapse:
  // https://www.tensorflow.org/xla/operation_semantics#collapse However we use
  // high to low dimension indices.
  //
  // Collapse replaces the given subset of the operand's dimensions by a single
  // dimension. The input arguments are an arbitrary array of type T and a
  // compile-time-constant vector of dimension indices. The dimension indices
  // must be an in-order (high to low dimension numbers), consecutive subset of
  // T's dimensions. Thus, {0, 1, 2}, {0, 1}, or {1, 2} are all valid dimension
  // sets, but {1, 0} or {0, 2} are not.
  mlir_dim_t clap_dims_size = inp_clap_dims.size();
  SmallValueVec4 claps_dim_values;
  if (clap_dims_size == 0) {
    return std::make_tuple(tensor, claps_dim_values);
  }

  // CHECK the input collapse dimensions are in-order, otherwise throw exception
  auto rank = GetRankOfMlirValue(tensor);
  SmallVec4<mlir_dim_t> clap_dims = NormalizeDimIndex(inp_clap_dims, rank);
  for (size_t k = 1; k < clap_dims_size; ++k) {
    MHLO_CHECK(clap_dims[k] == clap_dims[k - 1] + 1,
               "collapse dim not in consecutive order");
  }

  // get original tensor shape in mlir standard dialect
  auto shape_values = BuildDimSizeListOfTensor(builder, loc, tensor);

  // calculate the collapse new_dim, which build the graph in mlir standard
  // dialect
  for (auto k : clap_dims) {
    auto dim_size = shape_values[k];
    claps_dim_values.push_back(dim_size);
  }

  auto numel_prod = 1;
  for (auto dim_val : claps_dim_values) {
    if (auto dim_size = CastStdConstToI64(dim_val)) {
      numel_prod *= *dim_size;
    }
  }
  auto new_dim = BuildStdConstForI32(builder, loc, numel_prod);
  for (auto dim_val : claps_dim_values) {
    if (!CastStdConstToI64(dim_val)) {
      new_dim = builder.create<mlir::arith::MulIOp>(loc, new_dim, dim_val);
    }
  }

  // gather the new shape values
  SmallValueVec4 new_shape_vals;
  for (size_t k = 0; k < clap_dims[0]; ++k) {
    new_shape_vals.push_back(shape_values[k]);
  }
  new_shape_vals.push_back(new_dim);
  for (size_t k = clap_dims[clap_dims_size - 1] + 1; k < rank; ++k) {
    new_shape_vals.push_back(shape_values[k]);
  }

  return std::make_tuple(
      BuildDynamicReshapeTensor(builder, loc, tensor, new_shape_vals),
      claps_dim_values);
}

mlir::Value BuildExpandTensorShapeWithDhloDims(
    mlir::OpBuilder& builder, const mlir::Location& loc,
    const mlir::Value& tensor, const SmallValueVec4& expand_dims,
    mlir_dim_t expand_pos) {
  if (expand_dims.size() == 0) {
    return tensor;
  }

  auto shape_values = BuildDimSizeListOfTensor(builder, loc, tensor);
  mlir_dim_t rank = shape_values.size();
  mlir_dim_t new_rank = rank + expand_dims.size() - 1;
  expand_pos = NormalizeDimIndex(expand_pos, rank);

  SmallValueVec4 new_shape;
  for (mlir_dim_t k = 0; k < rank; ++k) {
    if (k == expand_pos) {
      new_shape.insert(new_shape.end(), expand_dims.begin(), expand_dims.end());
    } else {
      new_shape.push_back(shape_values[k]);
    }
  }

  return BuildDynamicReshapeTensor(builder, loc, tensor, new_shape);
}

mlir::Value BuildFromElements(mlir::OpBuilder& builder,
                              const mlir::Location& loc,
                              const SmallValueVec4& values) {
  mlir_dim_t len = static_cast<mlir_dim_t>(values.size());
  mlir::Value shape = builder.create<mlir::tensor::FromElementsOp>(loc, values);
  return shape;
}

mlir::Value BuildFromElements(mlir::OpBuilder& builder,
                              const mlir::Location& loc,
                              const mlir::Value& scalar) {
  // mlir::tensor::FromElementsOp doesn't support creation of rank
  // 0 tensor. To workaround, we first make an rank 1 tensor, then reshape to
  // rank 0.
  mlir::Value shape = builder.create<mlir::tensor::FromElementsOp>(loc, scalar);
  shape = BuildReshapeTensorToScalar(builder, loc, shape);
  return shape;
}

mlir::Value BuildPermute(mlir::OpBuilder& builder, const mlir::Location& loc,
                         const mlir::Value& input,
                         const SmallVec4<mlir_dim_t>& trans_dim_vec) {
  auto permutation_attr = BuildI64ElementsAttr(builder, trans_dim_vec);
  auto input_ranked_type = GetMilrRankedTensorType(input);
  SmallVec4<mlir_dim_t> ranked_shape(input_ranked_type.getRank(),
                                     mlir::ShapedType::kDynamicSize);
  auto mlir_tensor_type = mlir::RankedTensorType::get(
      ranked_shape, input_ranked_type.getElementType());
  auto result = builder.create<mlir::mhlo::TransposeOp>(
      loc, mlir_tensor_type, input, permutation_attr);
  return result.getResult();
}
}  // namespace mhlo
}  // namespace mlir

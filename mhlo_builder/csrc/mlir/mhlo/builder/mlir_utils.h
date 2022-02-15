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

#pragma once
#include "mlir/mhlo/builder/macros.h"
#include "mlir/mhlo/builder/mlir_type_utils.h"

namespace mlir {
namespace mhlo {

// Return the range indices [min, max)
SmallVec4<mlir_dim_t> RangeIndices(mlir_dim_t min, mlir_dim_t max);

mlir_dim_t GetRankOfMlirValue(const mlir::Value& tensor);

// Return the length if the tensor is a static length vector else -1
llvm::Optional<mlir_dim_t> GetLenIfStaticLenVector(const mlir::Value& tensor);

// The dimensions support "negative indexing", -1 would thus map to the last
// dimension, -2 to the preceding one, etc.
// NormalizeDimIndex would return the non-negative indexing.
mlir_dim_t NormalizeDimIndex(mlir_dim_t dim_size, mlir_dim_t rank);

// NormalizeDimIndex would return the non-negative indexing.
SmallVec4<mlir_dim_t> NormalizeDimIndex(const SmallVec4<mlir_dim_t>& dims,
                                        mlir_dim_t rank);

// Return the Number of Elements of the input Tensor in mlir standard dialect
mlir::Value BuildStandardI32NumelOfTensor(
    mlir::OpBuilder& builder, const mlir::Location& loc,
    const mlir::Value& input, const SmallVec4<mlir_dim_t>& numel_dims);

// Return the Number of Elements of the input Tensor in HLO dialect
mlir::Value BuildHloNumelOfTensor(mlir::OpBuilder& builder,
                                  const mlir::Location& loc,
                                  const mlir::Value& input,
                                  const SmallVec4<mlir_dim_t>& numel_dims);

// Returns minimal value for the given int or float element type.
mlir::Value BuildHloMinValueForType(mlir::OpBuilder& builder,
                                    const mlir::Location& loc,
                                    const mlir::Type& elem_type);

// Returns maximal value for the given int or float element type.
mlir::Value BuildHloMaxValueForType(mlir::OpBuilder& builder,
                                    const mlir::Location& loc,
                                    const mlir::Type& elem_type);

// Return int or float DenseElementsAttr of const 0
mlir::Value BuildHloConstZeroForType(mlir::OpBuilder& builder,
                                     const mlir::Location& loc,
                                     const mlir::Type& elem_type);

// Return int or float DenseElementsAttr of const 1
mlir::Value BuildHloConstOneForType(mlir::OpBuilder& builder,
                                    const mlir::Location& loc,
                                    const mlir::Type& elem_type);

// Returns float DenseElementsAttr of const value
mlir::Value BuildHloConstForFloatType(mlir::OpBuilder& builder,
                                      const mlir::Location& loc,
                                      const mlir::Type& elem_type,
                                      const float value);

// Return the index-th element of the mlir::Value that represent a vector
mlir::Value BuildExtractVectorElement(mlir::OpBuilder& builder,
                                      const mlir::Location& loc,
                                      const mlir::Value& vector_value,
                                      mlir_dim_t index);

// Return unknown dimension size of runtime shape. The unknown dimension size
// would be calculated at runtime by the subgraph was built.
mlir::Value BuildResolveUnknownDimSizeI32(mlir::OpBuilder& builder,
                                          const mlir::Location& loc,
                                          const mlir::Value& input,
                                          const SmallValueVec4& i32_dim_sizes);

mlir::Value BuildReshapeTensorToScalar(mlir::OpBuilder& builder,
                                       const mlir::Location& loc,
                                       const mlir::Value& single_value_input);

SmallVec4<mlir_dim_t> GetDimSizeListFromHloDimValList(
    const SmallValueVec4& dim_vals);
}  // namespace mhlo
}  // namespace mlir

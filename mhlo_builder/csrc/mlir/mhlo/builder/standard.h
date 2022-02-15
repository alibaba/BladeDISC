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
#include "mlir/mhlo/builder/mlir_type_utils.h"

namespace mlir {
namespace mhlo {
// NB: standard here means mlir standard dialect

// Build a standard bool constant op from bool value
mlir::Value BuildStdConstForBool(mlir::OpBuilder& builder,
                                 const mlir::Location& loc, bool value);

// Build a standard i32 constant op from int32_t value
mlir::Value BuildStdConstForI32(mlir::OpBuilder& builder,
                                const mlir::Location& loc, int32_t value);

// Build a standard i64 constant op from int64_t value
mlir::Value BuildStdConstForI64(mlir::OpBuilder& builder,
                                const mlir::Location& loc, int64_t value);

// Build a standard f64 constant op from double value
mlir::Value BuildStdConstForF64(mlir::OpBuilder& builder,
                                const mlir::Location& loc, double value);

// Build a standard index constant op from int64_t value
mlir::Value BuildStdConstForIndex(mlir::OpBuilder& builder,
                                  const mlir::Location& loc, int64_t value);

mlir::Value BuildStdConstLike(mlir::OpBuilder& builder,
                              const mlir::Location& loc, int64_t value,
                              mlir::Value other);

// Build a subgraph that return dim size as standard i64
mlir::Value BuildStdDimSizeOfTensor(mlir::OpBuilder& builder,
                                    const mlir::Location& loc,
                                    const mlir::Value& tensor,
                                    mlir_dim_t dim_index);

// Build a subgraph that return dim sizes as standard i64
SmallValueVec4 BuildStdDimSizeListOfTensor(
    mlir::OpBuilder& builder, const mlir::Location& loc,
    const mlir::Value& tensor, const SmallVec4<mlir_dim_t>& dims = {});

// Return value of max(std_lhs, std_rhs)
mlir::Value BuildStdMaximumSigned(mlir::OpBuilder& builder,
                                  const mlir::Location& loc,
                                  const mlir::Value& std_lhs,
                                  const mlir::Value& std_rhs);

// Return value of min(std_lhs, std_rhs)
mlir::Value BuildStdMinimumSigned(mlir::OpBuilder& builder,
                                  const mlir::Location& loc,
                                  const mlir::Value& std_lhs,
                                  const mlir::Value& std_rhs);

// Return value of std_lhs % std_rhs
mlir::Value BuildStdRemainderSigned(mlir::OpBuilder& builder,
                                    const mlir::Location& loc,
                                    const mlir::Value& std_lhs,
                                    const mlir::Value& std_rhs);

// Return value of std_lhs + std_rhs
mlir::Value BuildStdAddSigned(mlir::OpBuilder& builder,
                              const mlir::Location& loc,
                              const mlir::Value& std_lhs,
                              const mlir::Value& std_rhs);

// Return value of std_lhs - std_rhs
mlir::Value BuildStdSubSigned(mlir::OpBuilder& builder,
                              const mlir::Location& loc,
                              const mlir::Value& std_lhs,
                              const mlir::Value& std_rhs);

// Return value of std_lhs / std_rhs
mlir::Value BuildStdMulSigned(mlir::OpBuilder& builder,
                              const mlir::Location& loc,
                              const mlir::Value& std_lhs,
                              const mlir::Value& std_rhs);

// Return value of std_lhs / std_rhs
mlir::Value BuildStdDivSigned(mlir::OpBuilder& builder,
                              const mlir::Location& loc,
                              const mlir::Value& std_lhs,
                              const mlir::Value& std_rhs);

// Return value of - std_val
mlir::Value BuildStdNegtive(mlir::OpBuilder& builder, const mlir::Location& loc,
                            const mlir::Value& std_val);

// Build conversion from standard scalar to Tensor
mlir::Value BuildStdScalarToHloTensor(
    mlir::OpBuilder& builder, const mlir::Location& loc,
    const mlir::Value& std_scalar,
    const llvm::Optional<mlir::Type>& elem_type_opt = llvm::None);

// Build conversion from standard scalar(interger) to index
mlir::Value BuildStdScalarToIndexType(mlir::OpBuilder& builder,
                                      const mlir::Location& loc,
                                      const mlir::Value& dim_size);

// Build conversion from standard scalar(interger) to mhlo_dim_t(i32)
SmallValueVec4 BuildStdScalarToHloDimType(mlir::OpBuilder& builder,
                                          const mlir::Location& loc,
                                          const SmallValueVec4& dim_sizes);

// Build conversion from standard scalar vector to Tensor
mlir::Value BuildStdScalarToHloTensor(
    mlir::OpBuilder& builder, const mlir::Location& loc,
    const SmallValueVec4& values,
    const llvm::Optional<mlir::Type>& elem_type_opt = llvm::None);

mlir::Value BuildStdScalarFromHloTensor(mlir::OpBuilder& builder,
                                        const mlir::Location& loc,
                                        const mlir::Value& scalar_tensor);

}  // namespace mhlo
}  // namespace mlir

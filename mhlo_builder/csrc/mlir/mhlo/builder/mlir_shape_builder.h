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
mlir::Value BuildHloDimSizeOfTensor(mlir::OpBuilder& builder,
                                    const mlir::Location& loc,
                                    const mlir::Value& tensor,
                                    mlir_dim_t dim_index);

// Build an mlir subgraph that get the tensor's shape
SmallValueVec4 BuildDimSizeListOfTensor(mlir::OpBuilder& builder,
                                        const mlir::Location& loc,
                                        const mlir::Value& tensor,
                                        const SmallVec4<mlir_dim_t>& dims = {});

mlir::Value BuildShapeOfTensor(mlir::OpBuilder& builder,
                               const mlir::Location& loc,
                               const mlir::Value& tensor);

// Build an mlir subgraph that reshape the tensor
mlir::Value BuildDynamicReshapeTensor(mlir::OpBuilder& builder,
                                      const mlir::Location& loc,
                                      const mlir::Value& tensor,
                                      const SmallValueVec4& new_shape_vals);

// Build an mlir subgraph that returns a new tensor with
// dims of size 1 inserted at the specified position.
mlir::Value BuildUnsqueezeTensorShape(mlir::OpBuilder& builder,
                                      const mlir::Location& loc,
                                      const mlir::Value& tensor,
                                      const SmallVec4<mlir_dim_t>& unsqz_dims);

// Build an mlir subgraph that returns a new tensor with
// dims of size 1 removed from the specified position.
//
// NB: The squeezed dim_sizes are considered to be 1,
// otherwise the compilation behaviors are undefined.
mlir::Value BuildSqueezeTensorShape(mlir::OpBuilder& builder,
                                    const mlir::Location& loc,
                                    const mlir::Value& tensor,
                                    const SmallVec4<mlir_dim_t>& sqz_dims);

std::tuple<mlir::Value, SmallValueVec4> BuildCollapseTensorShape(
    mlir::OpBuilder& builder, const mlir::Location& loc,
    const mlir::Value& tensor, const SmallVec4<mlir_dim_t>& clap_dims);

mlir::Value BuildExpandTensorShapeWithDhloDims(
    mlir::OpBuilder& builder, const mlir::Location& loc,
    const mlir::Value& tensor, const SmallValueVec4& expand_dims,
    mlir_dim_t expand_pos);

mlir::Value BuildFromElements(mlir::OpBuilder& builder,
                              const mlir::Location& loc,
                              const SmallValueVec4& values);

mlir::Value BuildFromElements(mlir::OpBuilder& builder,
                              const mlir::Location& loc,
                              const mlir::Value& scalar);

mlir::Value BuildPermute(mlir::OpBuilder& builder, const mlir::Location& loc,
                         const mlir::Value& input,
                         const SmallVec4<mlir_dim_t>& trans_dim_vec);
}  // namespace mhlo
}  // namespace mlir

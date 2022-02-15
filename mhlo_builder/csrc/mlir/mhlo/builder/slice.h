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

// The direct usage of the function is not recommended, because
// start_index & end_index must be normalized before it is called.
// It's recommended to use BuildDynamicSlice.
mlir::Value BuildDynamicSliceInternal(mlir::OpBuilder& builder,
                                      const mlir::Location& loc,
                                      const mlir::Value& input,
                                      const mlir::Value& start_index,
                                      const mlir::Value& end_index,
                                      const mlir::Value& step,
                                      mlir_dim_t dim_index);

mlir::Value BuildDynamicSlice(mlir::OpBuilder& builder,
                              const mlir::Location& loc,
                              const mlir::Value& input,
                              const mlir::Value& start_index,
                              const mlir::Value& end_index,
                              const mlir::Value& step, mlir_dim_t dim_index);

mlir::Value BuildFromElements(mlir::OpBuilder& builder,
                              const mlir::Location& loc,
                              const SmallValueVec4& values);

mlir::Value BuildFromElements(mlir::OpBuilder& builder,
                              const mlir::Location& loc,
                              const mlir::Value& scalar);

std::tuple<mlir::Value, mlir::Value> BuildHalfSplit(mlir::OpBuilder& builder,
                                                    const mlir::Location& loc,
                                                    const mlir::Value& input,
                                                    mlir_dim_t dim_index);

mlir::Value BuildSelect(mlir::OpBuilder& builder, const mlir::Location& loc,
                        const mlir::Value& input,
                        const mlir::Value& select_index, mlir_dim_t dim_index);

mlir::Value BuildRoll(mlir::OpBuilder& builder, const mlir::Location& loc,
                      const mlir::Value& input, mlir_dim_t shift,
                      mlir_dim_t dim);
}  // namespace mhlo
}  // namespace mlir

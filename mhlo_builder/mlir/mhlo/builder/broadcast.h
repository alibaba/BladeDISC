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

mlir::Value BuildBroadcastScalarAsTensor(mlir::OpBuilder& builder,
                                         const mlir::Location& loc,
                                         const mlir::Value& scalar,
                                         const mlir::Value& tensor);

mlir::Value BuildBroadcastTensorAsOther(mlir::OpBuilder& builder,
                                        const mlir::Location& loc,
                                        const mlir::Value& tensor,
                                        const mlir::Value& other);

// Broadcast Tensor to shape with dims_size
mlir::Value BuildBroadcastTensorInDims(
    mlir::OpBuilder& builder, const mlir::Location& loc,
    const mlir::Value& tensor, const SmallValueVec4& dims_size,
    const SmallVec4<mlir_dim_t>& broadcast_dims);
}  // namespace mhlo
}  // namespace mlir
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
mlir::Value BuildStandardNorm(mlir::OpBuilder& builder,
                              const mlir::Location& loc,
                              const mlir::Value& input,
                              const llvm::Optional<mlir::Value>& var,
                              const llvm::Optional<mlir::Value>& mean,
                              double eps,
                              const SmallVec4<mlir_dim_t>& broadcast_dims);

mlir::Value BuildStandardNorm(mlir::OpBuilder& builder,
                              const mlir::Location& loc,
                              const mlir::Value& input, double eps,
                              mlir_dim_t reduced_last_dims);

// math: y = input * gamma + beta
// where gamma and beta's dims must match last certain number dims of input
mlir::Value BuildElemAffine(
    mlir::OpBuilder& builder, const mlir::Location& loc,
    const mlir::Value& input, const llvm::Optional<mlir::Value>& gamma,
    const llvm::Optional<mlir::Value>& beta,
    const llvm::Optional<SmallVec4<mlir_dim_t>>& broadcast_dims);

}  // namespace mhlo
}  // namespace mlir

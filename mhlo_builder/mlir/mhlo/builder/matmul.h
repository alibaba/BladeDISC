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

mlir::Value BuildDotProduct_bmm(mlir::OpBuilder& builder,
                                const mlir::Location& loc,
                                const mlir::Value& inp_lhs,
                                const mlir::Value& inp_rhs);

mlir::Value BuildDotProduct_mm(mlir::OpBuilder& builder,
                               const mlir::Location& loc,
                               const mlir::Value& inp_lhs,
                               const mlir::Value& inp_rhs);

mlir::Value BuildDotProduct_mv(mlir::OpBuilder& builder,
                               const mlir::Location& loc,
                               const mlir::Value& inp_lhs,
                               const mlir::Value& inp_rhs);
}  // namespace mhlo
}  // namespace mlir
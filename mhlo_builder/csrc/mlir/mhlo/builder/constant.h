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

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/mhlo/builder/macros.h"
#include "mlir/mhlo/builder/mlir_type_utils.h"

namespace mlir {
namespace mhlo {
// Return true if the mlir::Value it produced from a mlir::mhlo::ConstantOp
bool IsHloConstant(const mlir::Value&);
bool IsStdConstant(const mlir::Value&);
SmallVec4<int64_t> CastHloConstToListOfI64(const mlir::Value& value);

llvm::Optional<int64_t> CastStdConstToI64(const mlir::Value& val);
llvm::Optional<int64_t> CastHloConstToI64(const mlir::Value& val);
}  // namespace mhlo
}  // namespace mlir

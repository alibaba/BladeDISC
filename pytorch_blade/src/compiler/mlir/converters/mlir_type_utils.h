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
#include <c10/core/ScalarType.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>

namespace torch {
namespace jit {
class Value;
}
} // namespace torch

namespace torch {
namespace blade {

mlir::Type BuildMlirElemType(
    mlir::Builder& builder,
    c10::ScalarType scalar_type);

mlir::RankedTensorType BuildMlirRankedTensorType(
    mlir::OpBuilder& builder,
    const torch::jit::Value& value,
    bool static_shape = false);

::llvm::Optional<mlir::Value> BuildCastWithJitType(
    mlir::OpBuilder& builder,
    const mlir::Location& loc,
    const mlir::Value& value,
    const torch::jit::Value* dtype);
} // namespace blade
} // namespace torch

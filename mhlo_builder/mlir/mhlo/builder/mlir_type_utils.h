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

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>

namespace mlir {
namespace mhlo {
using mlir_dim_t = int64_t;
using mhlo_dim_t = int32_t;

template <class T>
using SmallVec4 = llvm::SmallVector<T, 4>;
using SmallValueVec4 = SmallVec4<mlir::Value>;

mlir::Type GetMlirTensorElemType(const mlir::Value& value);
mlir::RankedTensorType GetMilrRankedTensorType(const mlir::Value& tensor);
mlir::Type BuildMHloDimType(mlir::OpBuilder& builder);
mlir::Value TryBuildElementTypeCast(mlir::OpBuilder& builder,
                                    const mlir::Location& loc,
                                    const mlir::Value& ml_tensor,
                                    const mlir::Type& elem_type);

// Build a fully dynamic mlir::RankedTensorType with the same rank and dtype
// to the input tensor
mlir::RankedTensorType BuildRankedTensorTypeFrom(const mlir::Value& tensor);
}  // namespace mhlo
}  // namespace mlir

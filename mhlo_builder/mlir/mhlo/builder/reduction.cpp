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

#include "mlir/mhlo/builder/reduction.h"

#include "mlir/mhlo/builder/element_wise_binary.h"
#include "mlir/mhlo/builder/mlir_utils.h"

namespace mlir {
namespace mhlo {
// math: f(input, numel) = input / numel
mlir::Value BuildNumelReciprocal(mlir::OpBuilder& builder,
                                 const mlir::Location& loc,
                                 const mlir::Value& input,
                                 const mlir::Value& numel) {
  auto elem_type = GetMlirTensorElemType(input);
  mlir::Value num_elem_tensor =
      builder.create<mlir::mhlo::ConvertOp>(loc, numel, elem_type);
  auto result = BuildMlirBinaryOp<mlir::chlo::BroadcastDivOp>(
      builder, loc, input, num_elem_tensor, elem_type);
  return result;
}

template <>
mlir::Value BuildReductionInitValue<mlir::mhlo::MaxOp>(
    mlir::OpBuilder& builder, const mlir::Location& loc,
    const mlir::Type& elem_type) {
  return BuildHloMinValueForType(builder, loc, elem_type);
}

template <>
mlir::Value BuildReductionInitValue<mlir::mhlo::MinOp>(
    mlir::OpBuilder& builder, const mlir::Location& loc,
    const mlir::Type& elem_type) {
  return BuildHloMaxValueForType(builder, loc, elem_type);
}
}  // namespace mhlo
}  // namespace mlir

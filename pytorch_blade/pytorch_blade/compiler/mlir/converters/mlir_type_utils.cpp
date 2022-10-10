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

#include "pytorch_blade/compiler/mlir/converters/mlir_type_utils.h"

#include <mlir-hlo/Dialect/mhlo/IR/hlo_ops.h> // from tf repo
#include <mlir/mhlo/builder/mlir_type_utils.h>

#include <torch/script.h>

namespace torch {
namespace blade {
using namespace mlir::mhlo;

mlir::Type BuildMlirElemType(
    mlir::Builder& builder,
    c10::ScalarType scalar_type) {
  // TODO: add more types
  switch (scalar_type) {
    case c10::kHalf:
      return builder.getF16Type();
    case c10::kFloat:
      return builder.getF32Type();
    case c10::kDouble:
      return builder.getF64Type();
    case c10::kBool:
      return builder.getI1Type();
    case c10::kByte:
      return builder.getIntegerType(8, /*isSigned*/ false);
    case c10::kChar:
      return builder.getIntegerType(8);
    case c10::kShort:
      return builder.getIntegerType(16);
    case c10::kInt:
      return builder.getIntegerType(32);
    case c10::kLong:
      return builder.getIntegerType(64);
    default:
      TORCH_CHECK(false, scalar_type, " is not support");
  }
}

mlir::RankedTensorType BuildMlirRankedTensorType(
    mlir::OpBuilder& builder,
    const torch::jit::Value& value,
    bool static_shape) {
  auto tensor_type = value.type()->cast<TensorType>();
  TORCH_CHECK(tensor_type != nullptr);
  c10::optional<uint64_t> optional_rank = tensor_type->sizes().size();
  mlir_dim_t rank = 0;
  if (!optional_rank) {
    LOG(WARNING) << "The tensor rank is unknown";
  } else {
    rank = *optional_rank;
  }
  std::vector<mlir_dim_t> ranked_shape(rank, mlir::ShapedType::kDynamicSize);
  if (static_shape) {
    auto sizes = tensor_type->sizes().concrete_sizes();
    TORCH_CHECK(sizes, "The tensor has not concrete sizes");
    ranked_shape = *sizes;
  }

  auto scalar_type = tensor_type->scalarType();
  mlir::Type type = builder.getF32Type();
  if (scalar_type) {
    type = BuildMlirElemType(builder, *scalar_type);
  } else {
    LOG(WARNING) << "The tensor scalar type is unknown, default to float32";
  }
  auto mlir_tensor_type = mlir::RankedTensorType::get(ranked_shape, type);
  return mlir_tensor_type;
}

::llvm::Optional<mlir::Value> BuildCastWithJitType(
    mlir::OpBuilder& builder,
    const mlir::Location& loc,
    const mlir::Value& value,
    const torch::jit::Value* jit_dtype) {
  auto input_val = value;
  auto jit_dtype_ival = torch::jit::toIValue(jit_dtype);
  if (jit_dtype_ival && !jit_dtype_ival->isNone()) {
    if (!jit_dtype_ival->isInt()) {
      return ::llvm::None;
    }
    // ScalarType in jit::Graph is type of int
    c10::ScalarType dtype =
        static_cast<c10::ScalarType>(jit_dtype_ival->toInt());
    auto elem_type = BuildMlirElemType(builder, dtype);
    if (elem_type != GetMlirTensorElemType(input_val)) {
      input_val =
          builder.create<mlir::mhlo::ConvertOp>(loc, input_val, elem_type);
    }
  }
  return input_val;
}

} // namespace blade
} // namespace torch

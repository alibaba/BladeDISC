/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef DISC_DISC_UTIL_H_
#define DISC_DISC_UTIL_H_
#include <vector>

#include "llvm/ADT/StringRef.h"         // TF:llvm-project
#include "mlir/IR/Attributes.h"         // TF:llvm-project
#include "mlir/IR/Builders.h"           // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace disc_ral {

constexpr llvm::StringRef kDhloInputShapeAttr = "disc.input_shape";
constexpr llvm::StringRef kDhloInputValueAttr = "disc.input_value";

inline mlir::DenseIntElementsAttr GetI64ElementsAttr(ArrayRef<int64_t> values,
                                                     Builder* builder) {
  RankedTensorType ty = RankedTensorType::get(
      {static_cast<int64_t>(values.size())}, builder->getIntegerType(64));
  return mlir::DenseIntElementsAttr::get(ty, values);
}

inline std::vector<int64_t> ConvertDenseIntAttr(
    mlir::DenseIntElementsAttr attr) {
  auto values = attr.getValues<int64_t>();
  return {values.begin(), values.end()};
}

inline std::vector<int64_t> ConvertDenseIntAttr(
    llvm::Optional<mlir::DenseIntElementsAttr> attr) {
  if (!attr) return {};
  return ConvertDenseIntAttr(*attr);
}

inline mlir::DenseElementsAttr GetScalarOfType(Type ty, int64_t raw_value) {
  RankedTensorType scalar_ty = RankedTensorType::get({}, ty);

  if (auto float_ty = ty.dyn_cast<FloatType>()) {
    APFloat value(float_ty.getFloatSemantics(), raw_value);
    return mlir::DenseElementsAttr::get(scalar_ty, value);
  }
  auto int_ty = ty.cast<IntegerType>();
  APInt value(int_ty.getWidth(), static_cast<int64_t>(raw_value), true);
  return mlir::DenseElementsAttr::get(scalar_ty, value);
}

bool IsSmallBuffer(Value value);

bool IsSmallCpuBuffer(Value value);

bool IsSmallCpuAlloc(Value alloc);

bool IsOpWriteValue(Operation* op, Value value);

bool IsMemRefAliasOp(Operation* op);

}  // namespace disc_ral
}  // namespace mlir

#endif  // DISC_DISC_UTIL_H_

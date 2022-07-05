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
#include "tensorflow/compiler/mlir/disc/utils/cycle_detector.h"

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

Value getRootMemRef(Value memref);

bool isSameUnderlineBuffer(Value lhs, Value rhs);

// For mhlo.EinsumOp. TODO: merge into mhlo dialect
enum EquationVariable { kIsLhs, kIsRhs, kIsResult };
bool parseEinsumEquation(
    llvm::StringRef equation,
    llvm::SmallDenseMap<char, llvm::SmallDenseMap<EquationVariable, size_t>>&
        tokens,
    SmallVector<char>* lhs_original_tokens,
    SmallVector<char>* rhs_original_tokens,
    SmallVector<char>* result_original_tokens);

llvm::Optional<int32_t> TryMergeNode(GraphCycles* graph_cycles, int32_t a,
                                     int32_t b);

SmallVector<Value, 4> GetAllPossibleUsedValues(Operation* op);

// Returns true if the shape constraint IR is enabled.
bool useShapeConstraintIR();

// Returns true if `DISC_ENABLE_HORIZONTAL_FUSION` is true
bool useHorizontalFusion();

}  // namespace disc_ral
}  // namespace mlir

namespace llvm {

template <>
struct DenseMapInfo<mlir::disc_ral::EquationVariable> {
  using StorageInfo = DenseMapInfo<uint32_t>;
  static inline mlir::disc_ral::EquationVariable getEmptyKey() {
    return static_cast<mlir::disc_ral::EquationVariable>(
        StorageInfo::getEmptyKey());
  }
  static inline mlir::disc_ral::EquationVariable getTombstoneKey() {
    return static_cast<mlir::disc_ral::EquationVariable>(
        StorageInfo::getTombstoneKey());
  }
  static unsigned getHashValue(mlir::disc_ral::EquationVariable v) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(v));
  }
  static bool isEqual(mlir::disc_ral::EquationVariable lhs,
                      mlir::disc_ral::EquationVariable rhs) {
    return lhs == rhs;
  }
};

}  // namespace llvm

#endif  // DISC_DISC_UTIL_H_

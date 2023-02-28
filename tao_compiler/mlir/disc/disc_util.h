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

#include "llvm/ADT/StringRef.h"  // TF:llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"         // TF:llvm-project
#include "mlir/IR/Builders.h"           // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/disc/utils/cycle_detector.h"

namespace mlir {
namespace disc_ral {

constexpr llvm::StringRef kDhloInputShapeAttr = "disc.input_shape";
constexpr llvm::StringRef kDhloInputValueAttr = "disc.input_value";
constexpr llvm::StringRef kFuncEliminatedDeadArgumentsAttr = "disc.elimargs";
constexpr llvm::StringRef kFuncCompIntensFusionAttr = "disc.comp_intens_fusion";
constexpr llvm::StringRef kDynLibPathAttr = "disc.dyn_lib_path";

inline SmallVector<Value, 4> getDimSizesOfTensor(PatternRewriter& rewriter,
                                                 Operation* op, Value value) {
  auto value_ty = value.getType().dyn_cast<RankedTensorType>();

  auto loc = op->getLoc();
  auto rank = value_ty.getRank();
  // Get int vector [0, 1, ..., rank-1]
  SmallVector<Value, 4> dim_sizes;
  for (size_t d = 0; d < rank; ++d) {
    dim_sizes.emplace_back(rewriter.create<tensor::DimOp>(loc, value, d));
  }
  return dim_sizes;
}

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

inline std::vector<int64_t> ConvertArrayAttrToInt(mlir::ArrayAttr array_attr) {
  SmallVector<float, 4> values;
  values.reserve(array_attr.getValue().size());
  for (Attribute val : array_attr.getValue()) {
    values.push_back(static_cast<int64_t>(val.cast<IntegerAttr>().getInt()));
  }
  return {values.begin(), values.end()};
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

// Returns true if `DISC_ENABLE_HORIZONTAL_FUSION` is true.
bool useHorizontalFusion();

// Returns true if `DISC_ENABLE_TRANSFORM_SCHEDULE` is true.
bool useTransformSchedule();

// Returns true if `DISC_ENABLE_TRANSFORM_GEMM_EPILOGUE_FUSION` is true.
bool useTransformGEMMEpilogueFusionSchedule();

// Returns true if `DISC_FAKE_QUANT_TO_QUANT_AND_DEQUANT` is true
bool lowerFakeQuantToQuantAndDequant();

// Returns true if `DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL` is true.
bool isMemIntensiveOptExperimentalEnabled();

// Returns true if `DISC_ENABLE_STITCH` is true.
bool isStitchEnabled();

// Returns true if `DISC_ENABLE_COMPUTE_INTENSIVE_FUSE` is true.
bool isCompIntensFusionEnabled();

// Returns data users of the value and its aliases (e.g. memref.cast).
// Here non-data users means DimOp, DeallocOp and ShapeOfOp.
SmallVector<Operation*, 4> getValueUsers(Value v);

// Returns true if the underlying buffer of this memref is a const buffer.
bool isConstantMemRef(Value value);

DenseFPElementsAttr GetF32ElementsAttr(Attribute attr, Builder* builder);

DenseIntElementsAttr GetI64ElementsAttrForSeq(int start, int end,
                                              Builder* builder);

// Returns the number of operands that are supposed to be written.
// For some ops (e.g. lmhlo ops), some operands are the output memrefs
// Thus these operands are supposed to be updated.
int getNumResultOperands(Operation* op);

// Return size in bytes of shared memory per thread-block that does not hurt
// occupancy. It varies in different architectures. Currently, return 8192 for
// simplification.
int getShmemSizeBytesNotAffectOccupancy(int cc_major, int cc_minor);

// Return the element type of the result of given lmhlo op.
Type getLhloOpsElementType(Operation* op);

Value CastMemRefTo(OpBuilder& b, Location loc, Value from, Type toType,
                   ValueRange toShape);

Value createViewLike(OpBuilder& b, Location loc, Value from, Value to);

SmallVector<Value> getShapeValues(OpBuilder* b, Value memref);

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

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

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // TF:llvm-project
#include "mlir/IR/MLIRContext.h"            // TF:llvm-project
#include "tensorflow/compiler/mlir/disc/IR/hlo_disc_ops.h"

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_PLACEMENT_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_PLACEMENT_UTILS_H_

namespace mlir {
namespace placement_utils {
// Attrs for OP type
constexpr llvm::StringRef kDiscShapeCalcAttr = "disc.shape_op";

// Attrs for placement
constexpr llvm::StringRef kDiscPlaceAssignment = "disc.device";
constexpr llvm::StringRef kInputPlacementAttr = "input_placements";
constexpr llvm::StringRef kOutputPlacementAttr = "output_placements";
constexpr llvm::StringRef kCpu = "cpu";
constexpr llvm::StringRef kGpu = "gpu";
constexpr llvm::StringRef kConst = "const";

enum class PlacementType { CPU, GPU, Const };

PlacementType PlacementFromString(StringRef);
StringRef PlacementToString(PlacementType type);

using ShapeOperandList = SmallVector<int, 3>;

// return Op's shape calculation operand list.
ShapeOperandList getShapeCalcOperandList(Operation* op);

LogicalResult parseEntryFunctionInputPlacements(
    func::FuncOp main, bool default_on_gpu, SmallVectorImpl<StringRef>& out);

LogicalResult parseEntryFunctionOutputPlacements(
    func::FuncOp main, bool default_on_gpu, SmallVectorImpl<StringRef>& out);

LogicalResult parseEntryFunctionInputPlacements(
    func::FuncOp main, bool default_on_gpu,
    SmallVectorImpl<PlacementType>& out);

LogicalResult parseEntryFunctionOutputPlacements(
    func::FuncOp main, bool default_on_gpu,
    SmallVectorImpl<PlacementType>& out);

// If Op is placed on GPU
bool OnGpu(Operation* op);

// Return true if the Operation is placed on GPU
// The typical usage is for mhlo ops on tensor layer
bool isGpuMhlo(Operation* op);

// Return true if the MemRef is placed on GPU
bool isGpuMemRef(Value memref);

// Check Op if it is a mhlo Op.
inline bool isMhloDialect(Operation* op) {
  return (op->getDialect() ==
              op->getContext()->getLoadedDialect<mhlo::MhloDialect>() ||
          op->getDialect() ==
              op->getContext()->getLoadedDialect<mhlo_disc::MhloDiscDialect>());
}

inline bool isStdOnTensor(Operation* op) {
  return isa<arith::IndexCastOp>(op) &&
         op->getResult(0).getType().isa<RankedTensorType>();
}

inline bool isMhloOrStdOnTensor(Operation* op) {
  return isMhloDialect(op) || isStdOnTensor(op);
}

// Check Op if it is a tensor Op.
inline bool isTensorDialect(Operation* op) {
  return (op->getDialect() ==
          op->getContext()->getLoadedDialect<tensor::TensorDialect>());
}

// Check Op if it is a mhlo Op or tensor Op.
inline bool isMarkShapeCalcTargetOp(Operation* op) {
  return isTensorDialect(op) || isMhloDialect(op) || isStdOnTensor(op);
}

}  // namespace placement_utils
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_PLACEMENT_UTILS_H_

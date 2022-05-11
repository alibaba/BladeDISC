/* Copyright 2022 The BladeDISC Authors. All Rights Reserved.

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

// This file implements the logic to do some shape optimizations on tensor
// level.

#include "tensorflow/compiler/mlir/disc/transforms/disc_shape_optimization_utils.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/disc/IR/disc_shape_ops.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"

#undef LLVM_DEBUG

#define LLVM_DEBUG(x) (x)

namespace mlir {
namespace disc_ral {

using ::mlir::func::FuncOp;

int64_t getNextSymbolicDimUniqueId() {
  static int64_t id = 0;
  return id++;
}

int64_t SymbolicDim::uniqueId() const { return uniqueId_; }

// Merge two SymbolicDim if they are compatible.
LogicalResult SymbolicDim::Merge(SymbolicDim* other) {
  if (!isDynamic() && !other->isDynamic() &&
      getDimSize() != other->getDimSize())
    return failure();
  if (isDynamic()) dimSize_ = other->getDimSize();
  return success();
}

SymbolicDimMgr::SymbolicDimMgr(ModuleOp m) {
  // TODO
}

LogicalResult SymbolicDimMgr::load() {
  // TODO
  return success();
}

SymbolicDim* SymbolicDimMgr::newSymbolicDim() {
  symbolicDimStorage_.emplace_back(new SymbolicDim);
  SymbolicDim* symbol = symbolicDimStorage_.back().get();
  symbolDimUnionSet_[symbol] = symbol;
  return symbol;
}

SymbolicDim* SymbolicDimMgr::newConstantSymbolicDim(int64_t val) {
  auto it = constantSymbolicDimMap_.find(val);
  if (it == constantSymbolicDimMap_.end()) {
    it = constantSymbolicDimMap_.insert(std::make_pair(val, newSymbolicDim()))
             .first;
    it->second->setDimSize(val);
  }
  return it->second;
}

SymbolicDim* SymbolicDimMgr::getRootSymbolicDim(SymbolicDim* symbol) {
  SymbolicDim* current = symbol;
  while (symbolDimUnionSet_[current] != current)
    current = symbolDimUnionSet_[current];
  return current;
}

bool SymbolicDimMgr::isSymbolicDimEqual(SymbolicDim* lhs, SymbolicDim* rhs) {
  SymbolicDim* lhsRoot = getRootSymbolicDim(lhs);
  SymbolicDim* rhsRoot = getRootSymbolicDim(rhs);
  return lhsRoot == rhsRoot;
}

LogicalResult SymbolicDimMgr::mapSymbolicDimEqual(SymbolicDim* lhs,
                                                  SymbolicDim* rhs) {
  SymbolicDim* lhsRoot = getRootSymbolicDim(lhs);
  SymbolicDim* rhsRoot = getRootSymbolicDim(rhs);

  if (lhsRoot != rhsRoot) {
    if (lhsRoot->uniqueId() < rhsRoot->uniqueId()) {
      if (failed(lhsRoot->Merge(rhsRoot))) return failure();
      symbolDimUnionSet_[rhsRoot] = lhsRoot;
    } else {
      if (failed(rhsRoot->Merge(lhsRoot))) return failure();
      symbolDimUnionSet_[lhsRoot] = rhsRoot;
    }
  }
  return success();
}

LogicalResult SymbolicDimMgr::save() {
  // TODO
  return success();
}

SmallVector<SymbolicDim*> SymbolicDimMgr::getOrCreateSymbolicDimsForRankedValue(
    Value value) {
  // TODO: load existing symbols from the attribute attached on the tensor type
  SmallVector<SymbolicDim*> symbols;
  auto ty = value.getType().cast<RankedTensorType>();
  for (int64_t dim : ty.getShape()) {
    symbols.push_back(dim == ShapedType::kDynamicSize
                          ? newSymbolicDim()
                          : newConstantSymbolicDim(dim));
  }

  return symbols;
}

llvm::Optional<SmallVector<FlatSymbolRefAttr>> getRankedValueSymbolicDimRefs(
    Value value) {
  return {};
}

}  // namespace disc_ral
}  // namespace mlir

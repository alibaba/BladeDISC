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

int64_t SymbolicDim::uniqueId() const {
  // TODO
  return 0;
}

SymbolicDimMgr::SymbolicDimMgr(ModuleOp m) {
  // TODO
}

LogicalResult SymbolicDimMgr::load() {
  // TODO
  return success();
}

SymbolicDim* SymbolicDimMgr::newSymbolicDim() {
  // TODO
  return new SymbolicDim;
}

LogicalResult SymbolicDimMgr::save() {
  // TODO
  return success();
}

SmallVector<SymbolicDim*> SymbolicDimMgr::getOrCreateSymbolicDimsForRankedValue(
    Value value) {
  SmallVector<SymbolicDim*> symbols;
  return symbols;
}

llvm::Optional<SmallVector<FlatSymbolRefAttr>> getRankedValueSymbolicDimRefs(
    Value value) {
  return {};
}

}  // namespace disc_ral
}  // namespace mlir

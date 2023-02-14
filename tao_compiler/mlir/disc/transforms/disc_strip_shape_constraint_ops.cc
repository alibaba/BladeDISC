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

// This file implements logic to strip disc shape constraint ops.
#include "llvm/Support/Debug.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/disc/IR/disc_shape_ops.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/disc_shape_optimization_utils.h"

namespace mlir {
namespace disc_ral {

namespace {

struct DiscStripShapeConstraintOpsPass
    : public DiscStripShapeConstraintOpsPassBase<
          DiscStripShapeConstraintOpsPass> {
  void runOnOperation() override {
    SmallVector<disc_shape::SymbolicDimOp> ops;
    getOperation().walk(
        [&](disc_shape::SymbolicDimOp op) { ops.push_back(op); });
    for (disc_shape::SymbolicDimOp op : ops) {
      op->erase();
    }
    StringRef funcName = SymbolicDimMgr::getShapeConstraintGraphFunctionName();
    // first try to remove the old shape constraint graph
    if (auto func = getOperation().lookupSymbol<func::FuncOp>(funcName))
      func->erase();
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createStripShapeConstraintOpsPass() {
  return std::make_unique<DiscStripShapeConstraintOpsPass>();
}

}  // namespace disc_ral
}  // namespace mlir
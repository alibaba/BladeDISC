/* Copyright 2023 The BladeDISC Authors. All Rights Reserved.

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

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/fusion_utils.h"

// This file implements the logic to duplicate some lmhlo operations in order
// to enable more opportunities for fusion and reduce memory footprint.

namespace mlir {
namespace disc_ral {

namespace {

using func::FuncOp;
using namespace lmhlo;

lmhlo::ConstantOp getConstProducer(Value value) {
  for (Operation* user : value.getUsers()) {
    auto constOp = dyn_cast<lmhlo::ConstantOp>(user);
    if (constOp) return constOp;
  }
  return nullptr;
}

struct DiscDuplicateComputationAfterFusionPass
    : public DiscDuplicateComputationAfterFusionPassBase<
          DiscDuplicateComputationAfterFusionPass> {
  using DiscDuplicateComputationAfterFusionPassBase<
      DiscDuplicateComputationAfterFusionPass>::
      DiscDuplicateComputationAfterFusionPassBase;

  void runOnOperation() override {
    FuncOp func = getOperation();
    // skip shape constraint graph
    if (func.getName() == SymbolicDimMgr::getShapeConstraintGraphFunctionName())
      return;
    if (useTransformSchedule()) {
      if (failed(duplicateForTransformBaseFusion(func))) {
        return signalPassFailure();
      }
    }
  }

  LogicalResult duplicateForTransformBaseFusion(FuncOp func);
};

LogicalResult
DiscDuplicateComputationAfterFusionPass::duplicateForTransformBaseFusion(
    FuncOp func) {
  SmallVector<lmhlo::FusionOp> fusionOps;
  func->walk([&](lmhlo::FusionOp fusionOp) { fusionOps.push_back(fusionOp); });

  std::unique_ptr<ShapeAnalysis> shapeAnalysisPtr;
  if (useShapeConstraintIR()) {
    shapeAnalysisPtr.reset(new ShapeConstraintIRAnalysis(func));
  } else {
    shapeAnalysisPtr.reset(new ShapeAnalysisDeprecated{func});
    if (failed(static_cast<ShapeAnalysisDeprecated*>(shapeAnalysisPtr.get())
                   ->run())) {
      return failure();
    }
  }
  for (FusionOp fusionOp : fusionOps) {
    FusionPattern fusionPattern(fusionOp, shapeAnalysisPtr.get());
    if (!fusionPattern.isTransformBasedFusion()) continue;
    for (Value operand : fusionPattern.getOperands()) {
      auto constOp = getConstProducer(operand);
      if (!constOp ||
          constOp.getValue().cast<ElementsAttr>().getNumElements() != 1)
        continue;
      OpBuilder b(fusionOp);
      auto newBuffer = b.create<memref::AllocOp>(
          constOp->getLoc(),
          constOp->getOperand(0).getType().cast<MemRefType>());
      b.setInsertionPointToStart(fusionOp.getBody());
      auto newConstOp = b.clone(*constOp);
      newConstOp->setOperand(0, newBuffer);
      for (Operation* op : fusionPattern.getOpList()) {
        auto operands = op->getOperands();
        if (llvm::find(operands, constOp->getOperand(0)) == operands.end())
          continue;
        op->replaceUsesOfWith(constOp->getOperand(0), newBuffer);
      }
    }
  }

  return success();
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createDiscDuplicateComputationAfterFusionPass() {
  return std::make_unique<DiscDuplicateComputationAfterFusionPass>();
}

}  // namespace disc_ral
}  // namespace mlir

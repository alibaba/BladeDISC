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

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "tensorflow/compiler/mlir/disc/disc_util.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"
#include "tensorflow/compiler/mlir/disc/transforms/disc_shape_optimization_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/fusion_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/placement_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/shape_utils.h"

namespace mlir {
namespace disc_ral {

using func::FuncOp;
using namespace lmhlo;
using placement_utils::kDiscPlaceAssignment;

namespace {

struct DiscDuplicateComputationForFusionPass
    : public DiscDuplicateComputationForFusionPassBase<
          DiscDuplicateComputationForFusionPass> {
  using DiscDuplicateComputationForFusionPassBase<
      DiscDuplicateComputationForFusionPass>::
      DiscDuplicateComputationForFusionPassBase;

  explicit DiscDuplicateComputationForFusionPass(
      bool gpu_enabled, const std::string& fusion_strategy)
      : DiscDuplicateComputationForFusionPassBase<
            DiscDuplicateComputationForFusionPass>::
            DiscDuplicateComputationForFusionPassBase() {
    this->gpu_enabled_ = gpu_enabled;
    this->fusion_strategy_ = fusion_strategy;
  }

  void runOnOperation() override {
    FuncOp func = getOperation();

    // skip shape constraint graph
    if (func.getName() == SymbolicDimMgr::getShapeConstraintGraphFunctionName())
      return;

    // Note that we always use base strategy here. Thus the duplicated ops are
    // always supposed to be fused with other ops. It's a conservative strategy.
    // Re-visit this when necessary.
    auto strategy = makeNewPlacementAwareFusionStrategy(gpu_enabled_, "base");
    if (failed(duplicateBroadcastInDimOp(func, *strategy))) {
      signalPassFailure();
      return;
    }
  }

  LogicalResult duplicateBroadcastInDimOp(FuncOp func,
                                          FusionStrategy& strategy);
};

LogicalResult DiscDuplicateComputationForFusionPass::duplicateBroadcastInDimOp(
    FuncOp func, FusionStrategy& strategy) {
  SmallVector<Operation*> ops;
  func->walk([&](Operation* op) {
    if (isa<lmhlo::BroadcastOp, lmhlo::BroadcastInDimOp,
            lmhlo::DynamicBroadcastInDimOp>(op)) {
      ops.push_back(op);
    }
  });
  for (Operation* op : ops) {
    Value in = op->getOperand(0);
    Value out = cast<lmhlo::LmhloOp>(op).getResultBuffer();
    auto operandTy = in.getType().dyn_cast<MemRefType>();
    Operation* allocOp = out.getDefiningOp<memref::AllocOp>();
    if (!operandTy || operandTy.getRank() != 0 || !allocOp) continue;
    SmallVector<Operation*> fusibleUsers;
    for (Operation* user : out.getUsers()) {
      if (user == op) continue;
      if (strategy.isFusible(user)) fusibleUsers.push_back(user);
    }
    for (size_t i = 1; i < fusibleUsers.size(); ++i) {
      OpBuilder b(fusibleUsers[i]);
      Operation* clonedAllocOp = b.clone(*allocOp);
      Operation* clonedBcastOp = b.clone(*op);
      clonedBcastOp->replaceUsesOfWith(out, clonedAllocOp->getResult(0));
      fusibleUsers[i]->replaceUsesOfWith(out, clonedAllocOp->getResult(0));
    }
  }
  return success();
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createDiscDuplicateComputationForFusionPass(
    bool gpu_enabled, const std::string& fusion_strategy) {
  return std::make_unique<DiscDuplicateComputationForFusionPass>(
      gpu_enabled, fusion_strategy);
}

}  // namespace disc_ral
}  // namespace mlir

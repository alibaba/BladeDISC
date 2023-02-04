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

#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/disc/IR/lhlo_disc_ops.h"
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/PassDetail.h"

// This file implements logic to insert dealloc op for some disc specific ops
// (e.g. lmhlo_disc.custom_call_v2 op). These buffers can not be handled by
// normal deallocation pass, thus needs a dedicated pass.
//
// Currently we only support SCF, not CFG

namespace mlir {
namespace disc_ral {

namespace {

SmallVector<Value> findCandidates(Operation* op) {
  SmallVector<Value> candidates;
  op->walk([&](lmhlo_disc::CustomCallV2Op customCallOp) {
    for (Value v : customCallOp->getResults()) candidates.push_back(v);
  });
  return candidates;
}

LogicalResult insertDeallocOp(BufferViewFlowAnalysis& aliasAnalysis,
                              Liveness& livenessAnalysis,
                              PostDominanceInfo& postDominators, Value alloc) {
  auto aliasesSet = aliasAnalysis.resolve(alloc);
  assert(!aliasesSet.empty() && "must contain at least one alias");
  // Determine the actual block to place the dealloc and get liveness
  // information.
  Block* placementBlock =
      bufferization::BufferPlacementTransformationBase::findCommonDominator(
          alloc, aliasesSet, postDominators);
  const LivenessBlockInfo* livenessInfo =
      livenessAnalysis.getLiveness(placementBlock);

  // We have to ensure that the dealloc will be after the last use of all
  // aliases of the given value. We first assume that there are no uses in
  // the placementBlock and that we can safely place the dealloc at the
  // beginning.
  Operation* endOperation = &placementBlock->front();
  // Iterate over all aliases and ensure that the endOperation will point
  // to the last operation of all potential aliases in the placementBlock.
  for (Value alias : aliasesSet) {
    // Ensure that the start operation is at least the defining operation of
    // the current alias to avoid invalid placement of deallocs for aliases
    // without any uses.
    Operation* beforeOp = endOperation;
    if (alias.getDefiningOp() &&
        !(beforeOp =
              placementBlock->findAncestorOpInBlock(*alias.getDefiningOp())))
      continue;

    Operation* aliasEndOperation =
        livenessInfo->getEndOperation(alias, beforeOp);
    // Check whether the aliasEndOperation lies in the desired block and
    // whether it is behind the current endOperation. If yes, this will be
    // the new endOperation.
    if (aliasEndOperation->getBlock() == placementBlock &&
        endOperation->isBeforeInBlock(aliasEndOperation))
      endOperation = aliasEndOperation;
  }
  // endOperation is the last operation behind which we can safely store
  // the dealloc taking all potential aliases into account.

  // If the Dealloc position is at the terminator operation of the
  // block, then the value should escape from a deallocation.
  Operation* nextOp = endOperation->getNextNode();
  if (nextOp) {
    // If there is no dealloc node, insert one in the right place.
    OpBuilder builder(nextOp);
    builder.create<memref::DeallocOp>(alloc.getLoc(), alloc);
  }
  return success();
}

struct DiscBufferDeallocationPass
    : public DiscBufferDeallocationPassBase<DiscBufferDeallocationPass> {
  void runOnOperation() override {
    Operation* op = getOperation();
    BufferViewFlowAnalysis aliasAnalysis(op);
    Liveness livenessAnalysis(op);
    PostDominanceInfo postDominators(op);
    for (Value candidate : findCandidates(op)) {
      if (failed(insertDeallocOp(aliasAnalysis, livenessAnalysis,
                                 postDominators, candidate)))
        return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createDiscBufferDeallocationPass() {
  return std::make_unique<DiscBufferDeallocationPass>();
}

}  // namespace disc_ral
}  // namespace mlir

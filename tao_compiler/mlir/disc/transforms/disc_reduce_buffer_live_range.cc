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

#include "lhlo/IR/lhlo_ops.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/PassDetail.h"

namespace mlir {
namespace disc_ral {

using lmhlo::FusionOp;
using memref::AllocOp;

namespace {

LogicalResult moveBufferAllocator(AllocOp allocOp) {
  Value alloc = allocOp.getResult();
  BufferViewFlowAnalysis aliasAnalysis(allocOp);
  PostDominanceInfo postDominators(allocOp);
  auto aliasesSet = aliasAnalysis.resolve(alloc);
  // Determine the actual block to place the alloc and get liveness
  // information.
  Block* placementBlock =
      bufferization::BufferPlacementTransformationBase::findCommonDominator(
          alloc, aliasesSet, postDominators);

  Operation* toMoveBefore = nullptr;
  for (auto user : alloc.getUsers()) {
    if (isa<func::ReturnOp>(user)) continue;
    // user maybe in the sub-block of the placementBlock,
    // find the closest parent op inside of placementBlock
    while (user->getBlock() != placementBlock) {
      user = user->getParentOp();
    }
    if (toMoveBefore == nullptr || user->isBeforeInBlock(toMoveBefore)) {
      toMoveBefore = user;
    }
  }
  allocOp->moveBefore(toMoveBefore);
  return success();
}

struct DiscReduceBufferLiveRangePass
    : public DiscReduceBufferLiveRangePassBase<DiscReduceBufferLiveRangePass> {
  void runOnOperation() override {
    SmallVector<AllocOp, 4> candidateBuffers;
    func::FuncOp func = getOperation();

    func.walk([&](AllocOp op) { candidateBuffers.push_back(op); });

    for (int i = 0; i < candidateBuffers.size(); ++i) {
      if (failed(moveBufferAllocator(candidateBuffers[i]))) {
        return signalPassFailure();
      }
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createDiscReduceBufferLiveRangePass() {
  return std::make_unique<DiscReduceBufferLiveRangePass>();
}

}  // namespace disc_ral
}  // namespace mlir

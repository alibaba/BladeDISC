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

// This file implements the logic to create a parallel schedule for parallel ops
// on the cpu device.

#include "lhlo/IR/lhlo_ops.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/disc/IR/disc_ral_ops.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/codegen_utils.h"
#include "mlir/disc/transforms/fusion_utils.h"
#include "mlir/disc/transforms/placement_utils.h"

namespace mlir {
namespace disc_ral {

namespace {

const int64_t kMaxNumIterationsForSmallParallelOp = 64;
const int64_t kMaxNumIterationsForSmallForOp = 8;
const int64_t kVectorizationSize = 16;

bool isSmallParallelOp(scf::ParallelOp op) {
  int64_t numElems = 1;
  for (auto&& en :
       llvm::zip(op.getLowerBound(), op.getUpperBound(), op.getStep())) {
    auto lowerOp = dyn_cast_or_null<arith::ConstantIndexOp>(
        std::get<0>(en).getDefiningOp());
    auto upperOp = dyn_cast_or_null<arith::ConstantIndexOp>(
        std::get<1>(en).getDefiningOp());
    auto stepOp = dyn_cast_or_null<arith::ConstantIndexOp>(
        std::get<2>(en).getDefiningOp());
    if (!lowerOp || !upperOp || !stepOp) return false;
    auto step = stepOp.getValue().cast<IntegerAttr>().getInt();
    numElems *= ((upperOp.getValue().cast<IntegerAttr>().getInt() -
                  lowerOp.getValue().cast<IntegerAttr>().getInt() + step - 1) /
                 step);
  }
  return numElems < kMaxNumIterationsForSmallParallelOp;
}

bool isSmallForOp(scf::ForOp op) {
  auto lowerOp = dyn_cast_or_null<arith::ConstantIndexOp>(
      op.getLowerBound().getDefiningOp());
  auto upperOp = dyn_cast_or_null<arith::ConstantIndexOp>(
      op.getUpperBound().getDefiningOp());
  auto stepOp =
      dyn_cast_or_null<arith::ConstantIndexOp>(op.getStep().getDefiningOp());
  if (!lowerOp || !upperOp || !stepOp) return false;
  auto step = stepOp.getValue().cast<IntegerAttr>().getInt();
  int numIterations =
      ((upperOp.getValue().cast<IntegerAttr>().getInt() -
        lowerOp.getValue().cast<IntegerAttr>().getInt() + step - 1) /
       step);
  return numIterations < kMaxNumIterationsForSmallForOp;
}

bool isSmallCpuKernel(scf::ParallelOp op) {
  bool isSmall = true;
  op->walk([&](scf::ForOp forOp) {
    if (!isSmallForOp(forOp)) {
      isSmall = false;
    }
  });
  op->walk([&](scf::ParallelOp forOp) {
    if (!isSmallParallelOp(forOp)) {
      isSmall = false;
    }
  });
  return isSmall && isSmallParallelOp(op);
}

bool ParallelOpContainsSubLoops(scf::ParallelOp op) {
  bool hasSubLoops = false;
  op->walk([&](Operation* subLoop) {
    if (subLoop == op.getOperation()) return;
    if (isa<scf::ForOp, scf::ParallelOp>(subLoop)) hasSubLoops = true;
  });
  return hasSubLoops;
}

LogicalResult splitInnerMostParallelDim(OpBuilder& b, scf::ParallelOp op) {
  int numIVs = op.getLowerBound().size();
  assert(numIVs > 1);
  auto outter = b.create<scf::ParallelOp>(
      op.getLoc(), op.getLowerBound().drop_back(),
      op.getUpperBound().drop_back(), op.getStep().drop_back());
  b.setInsertionPointToStart(outter.getBody());
  auto inner = b.create<scf::ParallelOp>(
      op.getLoc(), op.getLowerBound().drop_front(numIVs - 1),
      op.getUpperBound().drop_front(numIVs - 1),
      op.getStep().drop_front(numIVs - 1));
  b.setInsertionPointToStart(inner.getBody());
  inner.getLoopBody().takeBody(op.getLoopBody());
  Block* entry = &inner.getLoopBody().front();
  for (auto&& en :
       llvm::zip(outter.getInductionVars(), entry->getArguments())) {
    std::get<1>(en).replaceAllUsesWith(std::get<0>(en));
  }
  while (entry->getNumArguments() > 1) {
    entry->eraseArgument(0);
  }
  op->erase();
  return success();
}

LogicalResult tileInnerMostParallelAxis(OpBuilder& b, scf::ParallelOp op) {
  tileParallelLoop(op, {kVectorizationSize}, /*withInboundCheck*/ true);
  return success();
}

struct DiscCpuMapParallelLoop
    : DiscCpuMapParallelLoopBase<DiscCpuMapParallelLoop> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    SmallVector<scf::ParallelOp> candidates;
    func.walk([&](scf::ParallelOp op) {
      if (op->getParentOfType<scf::ParallelOp>()) return;
      candidates.push_back(op);
    });

    for (scf::ParallelOp op : candidates) {
      if (failed(processParallelOp(op))) {
        signalPassFailure();
        return;
      }
    }
  }

  LogicalResult processParallelOp(scf::ParallelOp op);
};

LogicalResult DiscCpuMapParallelLoop::processParallelOp(scf::ParallelOp op) {
  if (op->getAttrOfType<UnitAttr>(kSmallCpuKernel)) return success();

  OpBuilder b(op);
  if (isSmallCpuKernel(op)) {
    op->setAttr(kSmallCpuKernel, b.getUnitAttr());
    return success();
  }

  return success();
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createDiscCpuMapParallelLoopPass() {
  return std::make_unique<DiscCpuMapParallelLoop>();
}

}  // namespace disc_ral
}  // namespace mlir

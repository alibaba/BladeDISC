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

#include "mlir-hlo/Dialect/mhlo/IR/disc_ral_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"
#include "tensorflow/compiler/mlir/disc/transforms/codegen_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/fusion_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/placement_utils.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace disc_ral {

namespace {

const int64_t kMaxNumIterationsForSmallParallelOp = 64;
const int64_t kMaxNumIterationsForSmallForOp = 8;
const int64_t kVectorizationSize = 16;

bool isSmallParallelOp(scf::ParallelOp op) {
  int64_t numElems = 1;
  for (auto &&en : llvm::zip(op.lowerBound(), op.upperBound(), op.step())) {
    auto lowerOp =
        dyn_cast_or_null<ConstantIndexOp>(std::get<0>(en).getDefiningOp());
    auto upperOp =
        dyn_cast_or_null<ConstantIndexOp>(std::get<1>(en).getDefiningOp());
    auto stepOp =
        dyn_cast_or_null<ConstantIndexOp>(std::get<2>(en).getDefiningOp());
    if (!lowerOp || !upperOp || !stepOp)
      return false;
    numElems *=
        ((upperOp.getValue() - lowerOp.getValue() + stepOp.getValue() - 1) /
         stepOp.getValue());
  }
  return numElems < kMaxNumIterationsForSmallParallelOp;
}

bool isSmallForOp(scf::ForOp op) {
  auto lowerOp =
      dyn_cast_or_null<ConstantIndexOp>(op.lowerBound().getDefiningOp());
  auto upperOp =
      dyn_cast_or_null<ConstantIndexOp>(op.upperBound().getDefiningOp());
  auto stepOp = dyn_cast_or_null<ConstantIndexOp>(op.step().getDefiningOp());
  if (!lowerOp || !upperOp || !stepOp)
    return false;
  int numIterations =
      ((upperOp.getValue() - lowerOp.getValue() + stepOp.getValue() - 1) /
       stepOp.getValue());
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
  op->walk([&](Operation *subLoop) {
    if (subLoop == op.getOperation())
      return;
    if (isa<scf::ForOp, scf::ParallelOp>(subLoop))
      hasSubLoops = true;
  });
  return hasSubLoops;
}

LogicalResult splitInnerMostParallelDim(OpBuilder &b, scf::ParallelOp op) {
  int numIVs = op.lowerBound().size();
  assert(numIVs > 1);
  auto outter = b.create<scf::ParallelOp>(
      op.getLoc(), op.lowerBound().drop_back(), op.upperBound().drop_back(),
      op.step().drop_back());
  b.setInsertionPointToStart(outter.getBody());
  auto inner = b.create<scf::ParallelOp>(
      op.getLoc(), op.lowerBound().drop_front(numIVs - 1),
      op.upperBound().drop_front(numIVs - 1), op.step().drop_front(numIVs - 1));
  b.setInsertionPointToStart(inner.getBody());
  inner.region().takeBody(op.region());
  Block *entry = &inner.region().front();
  for (auto &&en :
       llvm::zip(outter.getInductionVars(), entry->getArguments())) {
    std::get<1>(en).replaceAllUsesWith(std::get<0>(en));
  }
  while (entry->getNumArguments() > 1) {
    entry->eraseArgument(0);
  }
  op->erase();
  return success();
}

LogicalResult tileInnerMostParallelAxis(OpBuilder &b, scf::ParallelOp op) {
  tileParallelLoop(op, {kVectorizationSize}, /*withInboundCheck*/ true);
  return success();
}

struct DiscCpuMapParallelLoop
    : DiscCpuMapParallelLoopBase<DiscCpuMapParallelLoop> {
  void runOnFunction() override {
    FuncOp func = getFunction();
    SmallVector<scf::ParallelOp> candidates;
    func.walk([&](scf::ParallelOp op) {
      if (op->getParentOfType<scf::ParallelOp>())
        return;
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
  if (op->getAttrOfType<UnitAttr>(kSmallCpuKernel))
    return success();

  OpBuilder b(op);
  if (isSmallCpuKernel(op)) {
    op->setAttr(kSmallCpuKernel, b.getUnitAttr());
    return success();
  }

  return success();
}

} // namespace

std::unique_ptr<FunctionPass> createDiscCpuMapParallelLoopPass() {
  return std::make_unique<DiscCpuMapParallelLoop>();
}

} // namespace disc_ral
} // namespace mlir

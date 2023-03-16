// Copyright 2023 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/disc/tools/disc-transform/transforms/PassDetail.h"

#define DEBUG_TYPE "disc-convert-foreach-thread-op-to-parallel-op"

// This file implements the logic to convert scf.foreach_thread to scf.parallel

namespace mlir {
namespace disc_ral {
namespace {

using func::FuncOp;
using scf::ForeachThreadOp;
using scf::ParallelOp;

struct ForeachThreadToParallel : public OpRewritePattern<ForeachThreadOp> {
  using OpRewritePattern<ForeachThreadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForeachThreadOp foreachThreadOp,
                                PatternRewriter& rewriter) const override {
    if (foreachThreadOp->getNumResults() != 0) return failure();
    Location loc = foreachThreadOp.getLoc();
    int64_t rank = foreachThreadOp.getRank();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value> lowerBounds(rank, zero);
    SmallVector<Value> upperBounds = foreachThreadOp.getNumThreads();
    SmallVector<Value> steps(rank, one);

    auto parallelOp =
        rewriter.create<scf::ParallelOp>(loc, lowerBounds, upperBounds, steps);
    IRMapping mapping;
    for (const auto& z : llvm::zip(foreachThreadOp.getThreadIndices(),
                                   parallelOp.getInductionVars()))
      mapping.map(std::get<0>(z), std::get<1>(z));
    rewriter.setInsertionPointToStart(parallelOp.getBody());
    for (auto& nestedOp : foreachThreadOp.getBody()->without_terminator()) {
      rewriter.clone(nestedOp, mapping);
    }
    rewriter.replaceOp(foreachThreadOp, parallelOp->getResults());
    return success();
  }
};

struct DiscConvertForeachThreadOpToParallelOpPass
    : public DiscConvertForeachThreadOpToParallelOpPassBase<
          DiscConvertForeachThreadOpToParallelOpPass> {
  void runOnOperation() override;
};

void DiscConvertForeachThreadOpToParallelOpPass::runOnOperation() {
  MLIRContext* context = &getContext();
  RewritePatternSet patterns(&getContext());
  patterns.insert<ForeachThreadToParallel>(context);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createDiscConvertForeachThreadOpToParallelOpPass() {
  return std::make_unique<DiscConvertForeachThreadOpToParallelOpPass>();
}

}  // namespace disc_ral
}  // namespace mlir

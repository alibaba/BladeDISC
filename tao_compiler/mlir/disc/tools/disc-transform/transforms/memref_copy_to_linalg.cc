// Copyright 2022 The BladeDISC Authors. All rights reserved.
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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/disc/tools/disc-transform/transforms/PassDetail.h"
#include "mlir/disc/tools/disc-transform/utils.h"

#define DEBUG_TYPE "disc-memref-copy-to-linalg"

// This file implements the logic to convert a memref.copy op to its linalg
// equivalent.

namespace mlir {
namespace disc_ral {
namespace {

using func::FuncOp;

struct DiscMemrefCopyToLinalgPass
    : public DiscMemrefCopyToLinalgPassBase<DiscMemrefCopyToLinalgPass> {
  void runOnOperation() override;
};

struct MemrefCopyOpToLinalg : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter& rewriter) const override {
    Operation* linalgCopy =
        createLinalgCopyOp(rewriter, copyOp.getLoc(), copyOp.getSource(),
                           copyOp.getTarget(), copyOp->getAttrs());
    if (!linalgCopy) return failure();
    rewriter.replaceOp(copyOp, linalgCopy->getResults());
    return success();
  }
};

void DiscMemrefCopyToLinalgPass::runOnOperation() {
  MLIRContext* context = &getContext();
  RewritePatternSet patterns(&getContext());
  patterns.insert<MemrefCopyOpToLinalg>(context);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createDiscMemrefCopyToLinalgPass() {
  return std::make_unique<DiscMemrefCopyToLinalgPass>();
}

}  // namespace disc_ral
}  // namespace mlir

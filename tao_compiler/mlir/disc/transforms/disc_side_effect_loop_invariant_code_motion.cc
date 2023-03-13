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

// This pass makes `loop invariant code motion` optimization on operators with
// side effect.

#include "lhlo/IR/lhlo_ops.h"
#include "mlir/Dialect/SCF/IR/SCF.h"                     // from @llvm-project
#include "mlir/IR/Location.h"                            // from @llvm-project
#include "mlir/IR/MLIRContext.h"                         // from @llvm-project
#include "mlir/IR/Operation.h"                           // from @llvm-project
#include "mlir/IR/PatternMatch.h"                        // from @llvm-project
#include "mlir/Interfaces/LoopLikeInterface.h"           // from @llvm-project
#include "mlir/Pass/Pass.h"                              // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"                      // from @llvm-project
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/PassDetail.h"

namespace mlir {
namespace disc_ral {

namespace {

// Move the invariant memref::load op out of the loop.
template <typename LoopOp>
struct LoadLoopInvariantCodeMotion : public OpRewritePattern<LoopOp> {
  explicit LoadLoopInvariantCodeMotion(MLIRContext* context)
      : OpRewritePattern<LoopOp>(context) {}

  LogicalResult matchAndRewrite(LoopOp op,
                                PatternRewriter& rewriter) const override {
    LoopLikeOpInterface loop = dyn_cast<LoopLikeOpInterface>(op.getOperation());
    if (!loop) {
      return failure();
    }

    auto valueKnownNotModifiedInLoop = [&](Value value) {
      return llvm::all_of(value.getUsers(), [&](Operation* user) {
        // Check if `user` is outside of the loop.
        if (loop.getLoopBody().findAncestorOpInRegion(*user) == nullptr) {
          return true;
        }

        // Skip if there are memref alias operators.
        if (disc_ral::IsMemRefAliasOp(user)) {
          return false;
        }

        return !disc_ral::IsOpWriteValue(user, value);
      });
    };

    SmallVector<memref::LoadOp> toMoveOut;
    for (auto load : loop.getLoopBody().getOps<memref::LoadOp>()) {
      bool canMoveOut = true;

      // The inputs of `load` should be defined outside of the loop. The inputs
      // of `load` should not be modified inside the loop.
      auto input = load.getMemRef();
      canMoveOut &= loop.isDefinedOutsideOfLoop(input);
      canMoveOut &= valueKnownNotModifiedInLoop(input);
      for (auto index : load.getIndices()) {
        canMoveOut &= loop.isDefinedOutsideOfLoop(index);
        canMoveOut &= valueKnownNotModifiedInLoop(index);
      }

      if (canMoveOut) {
        toMoveOut.push_back(load);
      }
    }

    if (toMoveOut.empty()) {
      return failure();
    }

    for (auto load : toMoveOut) {
      loop.moveOutOfLoop(load);
    }

    return success();
  }
};

struct SideEffectLoopInvariantCodeMotionPass
    : public SideEffectLoopInvariantCodeMotionPassBase<
          SideEffectLoopInvariantCodeMotionPass> {
  void runOnOperation() override {
    gpu::GPUFuncOp func = getOperation();

    MLIRContext* ctx = func.getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<LoadLoopInvariantCodeMotion<scf::ForOp>,
                    LoadLoopInvariantCodeMotion<scf::WhileOp>,
                    LoadLoopInvariantCodeMotion<scf::ParallelOp>>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      func.emitError("applyPatternsAndFoldGreedily does not converge");
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<gpu::GPUFuncOp>>
createSideEffectLoopInvariantCodeMotionPass() {
  return std::make_unique<SideEffectLoopInvariantCodeMotionPass>();
}

}  // namespace disc_ral
}  // namespace mlir

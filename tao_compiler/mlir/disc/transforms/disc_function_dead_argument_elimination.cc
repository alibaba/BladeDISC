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

// This pass eliminates dead arguments of GPU LLVM functions.

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Location.h"                            // from @llvm-project
#include "mlir/IR/MLIRContext.h"                         // from @llvm-project
#include "mlir/IR/Operation.h"                           // from @llvm-project
#include "mlir/IR/PatternMatch.h"                        // from @llvm-project
#include "mlir/Pass/Pass.h"                              // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"                      // from @llvm-project
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/PassDetail.h"

namespace mlir {
namespace disc_ral {

namespace {

struct DeadArgumentElimination : public OpRewritePattern<LLVM::LLVMFuncOp> {
  explicit DeadArgumentElimination(MLIRContext* context)
      : OpRewritePattern(context) {}
  LogicalResult matchAndRewrite(LLVM::LLVMFuncOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<int64_t> deadArgsIndex;
    for (auto& en : llvm::enumerate(op.getBody().getArguments())) {
      Value argument = en.value();
      // Argument that is not used is regarded as dead argument.
      if (argument.use_empty()) {
        deadArgsIndex.push_back(en.index());
      }
    }

    if (deadArgsIndex.empty()) {
      return failure();
    }

    // The attribute will be used for lowering function call to ral logic.
    op->setAttr(kFuncEliminatedDeadArgumentsAttr,
                rewriter.getIndexArrayAttr(deadArgsIndex));

    // Remove the dead argument of the function body.
    for (auto arg : reverse(deadArgsIndex)) {
      op.getBody().eraseArgument(arg);
    }

    // To create new function prototype.
    LLVM::LLVMFunctionType origFuncType = op.getFunctionType();
    auto origInputs = origFuncType.getParams();
    SmallVector<Type> newInputs;
    for (int64_t i = 0; i < origInputs.size(); i++) {
      if (!llvm::is_contained(deadArgsIndex, i)) {
        newInputs.push_back(origInputs[i]);
      }
    }

    LLVM::LLVMFunctionType newType =
        origFuncType.clone(newInputs, origFuncType.getReturnTypes());
    auto newFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
        op.getLoc(), op.getName(), newType, op.getLinkage(), op.getDsoLocal(),
        op.getCConv(), op->getAttrs());

    // Move the body of original function into the new function.
    rewriter.inlineRegionBefore(op.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());

    rewriter.eraseOp(op);

    return success();
  }
};

struct FunctionDeadArgumentEliminationPass
    : public FunctionDeadArgumentEliminationPassBase<
          FunctionDeadArgumentEliminationPass> {
  void runOnOperation() override {
    gpu::GPUModuleOp m = getOperation();

    MLIRContext* ctx = m.getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<DeadArgumentElimination>(ctx);
    if (failed(applyPatternsAndFoldGreedily(m, std::move(patterns)))) {
      m.emitError("applyPatternsAndFoldGreedily does not converge");
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
createFunctionDeadArgumentEliminationPass() {
  return std::make_unique<FunctionDeadArgumentEliminationPass>();
}

}  // namespace disc_ral
}  // namespace mlir

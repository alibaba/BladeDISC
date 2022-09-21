// Copyright 2021 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/utils/hlo_utils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"       // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"                // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"               // from @llvm-project
#include "mlir/IR/Attributes.h"                          // from @llvm-project
#include "mlir/IR/Location.h"                            // from @llvm-project
#include "mlir/IR/MLIRContext.h"                         // from @llvm-project
#include "mlir/IR/Operation.h"                           // from @llvm-project
#include "mlir/IR/PatternMatch.h"                        // from @llvm-project
#include "mlir/Pass/Pass.h"                              // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"                      // from @llvm-project
#include "stablehlo/dialect/ChloOps.h"
#include "transforms/PassDetail.h"

namespace mlir {
namespace disc_ral {

namespace {
struct BatchNormInferenceOpConvert
    : public OpRewritePattern<mhlo::BatchNormInferenceOp> {
  explicit BatchNormInferenceOpConvert(MLIRContext* context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(mhlo::BatchNormInferenceOp op,
                                PatternRewriter& rewriter) const override {
    Value eps = chlo::getConstantLike(rewriter, op.getLoc(), op.epsilon(),
                                      op.variance());
    Value var = rewriter.create<mhlo::AddOp>(op.getLoc(), op.variance(), eps);
    Value rsqrt = rewriter.create<mhlo::RsqrtOp>(op.getLoc(), var);
    Value multiplier =
        rewriter.create<mhlo::MulOp>(op.getLoc(), rsqrt, op.scale());
    DenseIntElementsAttr broadcastDimensions =
        rewriter.getI64TensorAttr(ArrayRef<int64_t>{1});
    auto outType = op.getType();
    multiplier = rewriter.create<mhlo::BroadcastInDimOp>(
        op->getLoc(), outType, multiplier, broadcastDimensions);
    Value mean = rewriter.create<mhlo::BroadcastInDimOp>(
        op->getLoc(), outType, op.mean(), broadcastDimensions);
    Value offset = rewriter.create<mhlo::BroadcastInDimOp>(
        op->getLoc(), outType, op.offset(), broadcastDimensions);

    Value xSubMean =
        rewriter.create<mhlo::SubtractOp>(op.getLoc(), op.operand(), mean);
    Value comp1 =
        rewriter.create<mhlo::MulOp>(op.getLoc(), xSubMean, multiplier);
    rewriter.replaceOpWithNewOp<mhlo::AddOp>(op, comp1, offset);
    return success();
  }
};

struct MhloRewriterPass : public MhloRewriterPassBase<MhloRewriterPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext* ctx = func.getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<BatchNormInferenceOpConvert>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      func.emitError("applyPatternsAndFoldGreedily does not converge");
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createDiscMhloRewriterPass() {
  return std::make_unique<MhloRewriterPass>();
}

}  // namespace disc_ral
}  // namespace mlir

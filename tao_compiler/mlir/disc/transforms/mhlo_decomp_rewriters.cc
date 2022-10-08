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
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/disc/disc_util.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"

namespace mlir {
namespace disc_ral {

namespace {
Value broadcastAs(PatternRewriter& rewriter, Operation* op, Value self,
                  Value ref, DenseIntElementsAttr broadcastAttr) {
  auto loc = op->getLoc();
  auto dim_sizes = getDimSizesOfTensor(rewriter, op, ref);
  if (dim_sizes.size() == 0) {
    return rewriter.create<mhlo::ReshapeOp>(loc, ref.getType(), self);
  }
  Value shape = rewriter.create<tensor::FromElementsOp>(loc, dim_sizes);
  return rewriter.create<mhlo::DynamicBroadcastInDimOp>(
      loc, ref.getType(), self, shape, broadcastAttr);
}

Value getConstantLike(PatternRewriter& rewriter, Operation* op,
                      Attribute fillVal, Value ref) {
  Value val = rewriter.create<mhlo::ConstantOp>(op->getLoc(), fillVal);
  return broadcastAs(rewriter, op, val, ref, rewriter.getI64TensorAttr({}));
}

struct BatchNormInferenceOpConvert
    : public OpRewritePattern<mhlo::BatchNormInferenceOp> {
  explicit BatchNormInferenceOpConvert(MLIRContext* context)
      : OpRewritePattern(context) {}
  LogicalResult matchAndRewrite(mhlo::BatchNormInferenceOp op,
                                PatternRewriter& rewriter) const override;
};

LogicalResult BatchNormInferenceOpConvert::matchAndRewrite(
    mhlo::BatchNormInferenceOp op, PatternRewriter& rewriter) const {
  auto loc = op.getLoc();
  Value var = op.variance();
  Value eps = getConstantLike(rewriter, op, op.epsilonAttr(), var);
  var = rewriter.create<mhlo::AddOp>(loc, var, eps);

  Value rsqrt = rewriter.create<mhlo::RsqrtOp>(loc, var);
  Value multiplier = rewriter.create<mhlo::MulOp>(loc, rsqrt, op.scale());
  DenseIntElementsAttr broadcastDimensions =
      rewriter.getI64TensorAttr(ArrayRef<int64_t>{op.feature_index()});
  auto outType = op.getType();
  Value scale =
      broadcastAs(rewriter, op, multiplier, op.operand(), broadcastDimensions);
  Value mean =
      broadcastAs(rewriter, op, op.mean(), op.operand(), broadcastDimensions);
  Value offset =
      broadcastAs(rewriter, op, op.offset(), op.operand(), broadcastDimensions);

  Value xSubMean = rewriter.create<mhlo::SubtractOp>(loc, op.operand(), mean);
  Value comp1 = rewriter.create<mhlo::MulOp>(loc, xSubMean, scale);
  rewriter.replaceOpWithNewOp<mhlo::AddOp>(op, op.getType(), comp1, offset);
  return success();
}

struct MhloDecompositionRewriterPass
    : public MhloDecompositionRewriterPassBase<MhloDecompositionRewriterPass> {
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

std::unique_ptr<OperationPass<func::FuncOp>>
createDiscMhloDecompositionRewriterPass() {
  return std::make_unique<MhloDecompositionRewriterPass>();
}

}  // namespace disc_ral
}  // namespace mlir

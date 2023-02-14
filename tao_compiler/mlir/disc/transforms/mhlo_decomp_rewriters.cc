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

#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/PassDetail.h"

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
  Value var = op.getVariance();
  Value eps = getConstantLike(rewriter, op, op.getEpsilonAttr(), var);
  var = rewriter.create<mhlo::AddOp>(loc, var, eps);

  Value rsqrt = rewriter.create<mhlo::RsqrtOp>(loc, var);
  Value multiplier = rewriter.create<mhlo::MulOp>(loc, rsqrt, op.getScale());
  DenseIntElementsAttr broadcastDimensions =
      rewriter.getI64TensorAttr(ArrayRef<int64_t>{op.getFeatureIndex()});
  auto outType = op.getType();
  Value scale = broadcastAs(rewriter, op, multiplier, op.getOperand(),
                            broadcastDimensions);
  Value mean = broadcastAs(rewriter, op, op.getMean(), op.getOperand(),
                           broadcastDimensions);
  Value offset = broadcastAs(rewriter, op, op.getOffset(), op.getOperand(),
                             broadcastDimensions);

  Value xSubMean =
      rewriter.create<mhlo::SubtractOp>(loc, op.getOperand(), mean);
  Value comp1 = rewriter.create<mhlo::MulOp>(loc, xSubMean, scale);
  rewriter.replaceOpWithNewOp<mhlo::AddOp>(op, op.getType(), comp1, offset);
  return success();
}
}  // namespace

namespace {
struct PadOpConvert : public OpRewritePattern<mhlo::PadOp> {
  explicit PadOpConvert(MLIRContext* context) : OpRewritePattern(context) {}
  LogicalResult matchAndRewrite(mhlo::PadOp op,
                                PatternRewriter& rewriter) const override;
};

LogicalResult PadOpConvert::matchAndRewrite(mhlo::PadOp op,
                                            PatternRewriter& rewriter) const {
  auto loc = op.getLoc();
  Value paddingLowTensor =
      rewriter.create<mhlo::ConstantOp>(loc, op.getEdgePaddingLow());
  Value paddingHighTensor =
      rewriter.create<mhlo::ConstantOp>(loc, op.getEdgePaddingHigh());
  Value paddingIterTensor =
      rewriter.create<mhlo::ConstantOp>(loc, op.getInteriorPadding());
  auto operand = op.getOperand();
  auto padVal = op.getPaddingValue();
  rewriter.replaceOpWithNewOp<mhlo::DynamicPadOp>(
      op, op.getType(), operand, padVal, paddingLowTensor, paddingHighTensor,
      paddingIterTensor);
  return success();
}
}  // namespace

namespace {
struct SliceOpConvert : public OpRewritePattern<mhlo::SliceOp> {
  explicit SliceOpConvert(MLIRContext* context) : OpRewritePattern(context) {}
  LogicalResult matchAndRewrite(mhlo::SliceOp op,
                                PatternRewriter& rewriter) const override;
};

LogicalResult SliceOpConvert::matchAndRewrite(mhlo::SliceOp op,
                                              PatternRewriter& rewriter) const {
  auto loc = op.getLoc();
  Value startIndices =
      rewriter.create<mhlo::ConstantOp>(loc, op.getStartIndices());
  Value limitIndices =
      rewriter.create<mhlo::ConstantOp>(loc, op.getLimitIndices());
  Value strides = rewriter.create<mhlo::ConstantOp>(loc, op.getStrides());
  auto operand = op.getOperand();
  rewriter.replaceOpWithNewOp<mhlo::RealDynamicSliceOp>(
      op, op.getType(), operand, startIndices, limitIndices, strides);
  return success();
}
}  // namespace

struct MhloDecompositionRewriterPass
    : public MhloDecompositionRewriterPassBase<MhloDecompositionRewriterPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext* ctx = func.getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<BatchNormInferenceOpConvert>(ctx);
    patterns.insert<PadOpConvert>(ctx);
    patterns.insert<SliceOpConvert>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      func.emitError("applyPatternsAndFoldGreedily does not converge");
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<func::FuncOp>>
createDiscMhloDecompositionRewriterPass() {
  return std::make_unique<MhloDecompositionRewriterPass>();
}

}  // namespace disc_ral
}  // namespace mlir

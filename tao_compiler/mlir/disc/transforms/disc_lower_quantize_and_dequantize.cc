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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/disc/IR/hlo_disc_ops.h"
#include "mlir/disc/transforms/PassDetail.h"

#define DEBUG_TYPE "disc-lower-quantize-and-dequantize"

// This file implements the logic to decompose the quantize and dequantize op
// to a bunch of basic elementwise ops.

namespace mlir {
namespace disc_ral {
namespace {

struct QuantizeOpConverter : public OpRewritePattern<mhlo_disc::QuantizeOp> {
  using OpRewritePattern<mhlo_disc::QuantizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo_disc::QuantizeOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getUseDynamic())
      return rewriter.notifyMatchFailure(op,
                                         "Not support dynamic quantize a.t.m.");

    auto inputTy = op.getInput().getType().cast<RankedTensorType>();
    if (!inputTy.getElementType().isF32())
      return rewriter.notifyMatchFailure(
          op, "Only support quantize f32 input a.t.m.");

    Location loc = op.getLoc();
    Value inputShape = rewriter.create<shape::ShapeOfOp>(loc, op.getInput());
    Value bcastedScale = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc, inputTy, op.getScale(), inputShape, op.getAxis());
    auto zeroPointTy = op.getZeroPoint().getType().cast<RankedTensorType>();
    auto castedZeroPointTy =
        RankedTensorType::get(zeroPointTy.getShape(), inputTy.getElementType(),
                              zeroPointTy.getEncoding());
    Value castedZeroPoint = rewriter.create<mhlo::ConvertOp>(
        loc, castedZeroPointTy, op.getZeroPoint());
    Value bcastedZeroPoint = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc, inputTy, castedZeroPoint, inputShape, op.getAxis());
    auto quantMinMaxTy = RankedTensorType::get({}, inputTy.getElementType());
    Value quantMin = rewriter.create<mhlo::ConstantOp>(
        loc, DenseElementsAttr::get(
                 quantMinMaxTy,
                 static_cast<float>(static_cast<int64_t>(op.getQuantMin()))));
    Value quantMax = rewriter.create<mhlo::ConstantOp>(
        loc, DenseElementsAttr::get(
                 quantMinMaxTy,
                 static_cast<float>(static_cast<int64_t>(op.getQuantMax()))));
    // quantMin/Max should always be scalar.
    Value bcastedQuantMin = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc, inputTy, quantMin, inputShape, rewriter.getI64TensorAttr({}));
    Value bcastedQuantMax = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc, inputTy, quantMax, inputShape, rewriter.getI64TensorAttr({}));

    // output = clip(round(\frac{input}{scale} + zero\_point), quant\_min,
    // quant\_max)
    Value t0 = rewriter.create<mhlo::DivOp>(loc, op.getInput(), bcastedScale);
    Value t1 = rewriter.create<mhlo::AddOp>(loc, t0, bcastedZeroPoint);
    Value t2;
    if (op.getRoundMode() == mlir::mhlo_disc::RoundModeEnum::RoundHalfToEven) {
      t2 = rewriter.create<mhlo::RoundNearestEvenOp>(loc, t1);
    } else if (op.getRoundMode() ==
               mlir::mhlo_disc::RoundModeEnum::RoundHalfAwayFromZero) {
      t2 = rewriter.create<mhlo::RoundOp>(loc, t1);
    } else {
      return rewriter.notifyMatchFailure(op, "Round mode is not supported");
    }

    Value t3 = rewriter.create<mhlo::ClampOp>(loc, t2, bcastedQuantMin,
                                              bcastedQuantMax);
    Value out = rewriter.create<mhlo::ConvertOp>(loc, op.getType(), t3);
    rewriter.replaceOp(op, out);
    return success();
  }
};

struct DequantizeOpConverter
    : public OpRewritePattern<mhlo_disc::DequantizeOp> {
  using OpRewritePattern<mhlo_disc::DequantizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo_disc::DequantizeOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getUseDynamic())
      return rewriter.notifyMatchFailure(
          op, "Not support dynamic dequantize a.t.m.");

    auto outTy = op.getType().cast<RankedTensorType>();
    auto scaleTy = op.getScale().getType().cast<RankedTensorType>();
    if (outTy.getElementType() != scaleTy.getElementType() ||
        !outTy.getElementType().isF32())
      return rewriter.notifyMatchFailure(
          op, "Only support dequantize to f32 a.t.m.");

    Location loc = op.getLoc();
    auto inputTy = op.getInput().getType().cast<RankedTensorType>();
    Value inputShape = rewriter.create<shape::ShapeOfOp>(loc, op.getInput());
    Value bcastedScale = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc, outTy, op.getScale(), inputShape, op.getAxis());
    auto zeroPointTy = op.getZeroPoint().getType().cast<RankedTensorType>();
    auto bcastedInputOrZeroPointTy =
        RankedTensorType::get(inputTy.getShape(), zeroPointTy.getElementType(),
                              zeroPointTy.getEncoding());
    Value bcastedZeroPoint = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc, bcastedInputOrZeroPointTy, op.getZeroPoint(), inputShape,
        op.getAxis());
    Value castedInput = rewriter.create<mhlo::ConvertOp>(
        loc, bcastedInputOrZeroPointTy, op.getInput());

    // output = (input - zero\_point) \times scale
    Value t0 =
        rewriter.create<mhlo::SubtractOp>(loc, castedInput, bcastedZeroPoint);
    Value t1 = rewriter.create<mhlo::ConvertOp>(loc, outTy, t0);
    Value t2 = rewriter.create<mhlo::MulOp>(loc, t1, bcastedScale);
    rewriter.replaceOp(op, t2);
    return success();
  }
};

void populateQuantizeAndDequantizePatterns(RewritePatternSet& patterns) {
  // clang-format off
  patterns.insert<
    DequantizeOpConverter,
    QuantizeOpConverter
  >(patterns.getContext());
  // clang-format on
}

struct DiscLowerQuantizeAndDequantizePass
    : public DiscLowerQuantizeAndDequantizePassBase<
          DiscLowerQuantizeAndDequantizePass> {
  void runOnOperation() override;
};

void DiscLowerQuantizeAndDequantizePass::runOnOperation() {
  // Setup rewriter patterns.
  MLIRContext& ctx = getContext();
  RewritePatternSet patterns(&ctx);
  populateQuantizeAndDequantizePatterns(patterns);

  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
    return;
  }
}

struct GpuDequantizeOpConverter
    : public OpRewritePattern<mhlo_disc::DequantizeOp> {
  using OpRewritePattern<mhlo_disc::DequantizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo_disc::DequantizeOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getUseDynamic())
      return rewriter.notifyMatchFailure(
          op, "Not support dynamic dequantize a.t.m.");

    auto outTy = op.getType().cast<RankedTensorType>();
    auto scaleTy = op.getScale().getType().cast<RankedTensorType>();
    if (outTy.getElementType() != scaleTy.getElementType() ||
        !outTy.getElementType().isF32())
      return rewriter.notifyMatchFailure(
          op, "Only support dequantize to f32 a.t.m.");

    Location loc = op.getLoc();
    auto inputTy = op.getInput().getType().cast<RankedTensorType>();
    Value inputShape = rewriter.create<shape::ShapeOfOp>(loc, op.getInput());
    Value bcastedScale = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc, outTy, op.getScale(), inputShape, op.getAxis());
    auto zeroPointTy = op.getZeroPoint().getType().cast<RankedTensorType>();
    auto bcastedInputOrZeroPointTy = RankedTensorType::get(
        inputTy.getShape(), scaleTy.getElementType(), scaleTy.getEncoding());
    // Cast zeropoint from int32 to float32 first since int32 value would be
    // placed in host which will introduce many extra h2d and d2h overhead.
    Value castedZeroPoint =
        rewriter.create<mhlo::ConvertOp>(loc, scaleTy, op.getZeroPoint());
    Value bcastedZeroPoint = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc, bcastedInputOrZeroPointTy, castedZeroPoint, inputShape,
        op.getAxis());
    Value castedInput = rewriter.create<mhlo::ConvertOp>(
        loc, bcastedInputOrZeroPointTy, op.getInput());

    // output = (input - zero\_point) \times scale
    Value t0 =
        rewriter.create<mhlo::SubtractOp>(loc, castedInput, bcastedZeroPoint);
    Value t1 = rewriter.create<mhlo::ConvertOp>(loc, outTy, t0);
    Value t2 = rewriter.create<mhlo::MulOp>(loc, t1, bcastedScale);
    rewriter.replaceOp(op, t2);
    return success();
  }
};

void populateGpuQuantizeAndDequantizePatterns(RewritePatternSet& patterns) {
  // clang-format off
  patterns.insert<
    GpuDequantizeOpConverter,
    QuantizeOpConverter
  >(patterns.getContext());
  // clang-format on
}

struct DiscLowerGpuQuantizeAndDequantizePass
    : public DiscLowerGpuQuantizeAndDequantizePassBase<
          DiscLowerGpuQuantizeAndDequantizePass> {
  void runOnOperation() override;
};

void DiscLowerGpuQuantizeAndDequantizePass::runOnOperation() {
  // Setup rewriter patterns.
  MLIRContext& ctx = getContext();
  RewritePatternSet patterns(&ctx);
  populateGpuQuantizeAndDequantizePatterns(patterns);

  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
    return;
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createDiscLowerQuantizeAndDequantizePass() {
  return std::make_unique<DiscLowerQuantizeAndDequantizePass>();
}

std::unique_ptr<OperationPass<func::FuncOp>>
createDiscLowerGpuQuantizeAndDequantizePass() {
  return std::make_unique<DiscLowerGpuQuantizeAndDequantizePass>();
}

}  // namespace disc_ral
}  // namespace mlir

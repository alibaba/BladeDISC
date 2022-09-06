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
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/disc/IR/hlo_disc_ops.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"

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
    if (op.use_dynamic())
      return rewriter.notifyMatchFailure(op,
                                         "Not support dynamic quantize a.t.m.");

    auto inputTy = op.input().getType().cast<RankedTensorType>();
    if (!inputTy.getElementType().isF32())
      return rewriter.notifyMatchFailure(
          op, "Only support quantize f32 input a.t.m.");

    Location loc = op.getLoc();
    Value inputShape = rewriter.create<shape::ShapeOfOp>(loc, op.input());
    Value bcastedScale = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc, inputTy, op.scale(), inputShape, op.axis());
    auto zeroPointTy = op.zero_point().getType().cast<RankedTensorType>();
    auto castedZeroPointTy =
        RankedTensorType::get(zeroPointTy.getShape(), inputTy.getElementType(),
                              zeroPointTy.getEncoding());
    Value castedZeroPoint = rewriter.create<mhlo::ConvertOp>(
        loc, castedZeroPointTy, op.zero_point());
    Value bcastedZeroPoint = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc, inputTy, castedZeroPoint, inputShape, op.axis());
    auto quantMinMaxTy = RankedTensorType::get({}, inputTy.getElementType());
    Value quantMin = rewriter.create<mhlo::ConstantOp>(
        loc, DenseElementsAttr::get(
                 quantMinMaxTy,
                 static_cast<float>(static_cast<int64_t>(op.quant_min()))));
    Value quantMax = rewriter.create<mhlo::ConstantOp>(
        loc, DenseElementsAttr::get(
                 quantMinMaxTy,
                 static_cast<float>(static_cast<int64_t>(op.quant_max()))));
    // quantMin/Max should always be scalar.
    Value bcastedQuantMin = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc, inputTy, quantMin, inputShape, rewriter.getI64TensorAttr({}));
    Value bcastedQuantMax = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc, inputTy, quantMax, inputShape, rewriter.getI64TensorAttr({}));

    // output = clip(round(\frac{input}{scale} + zero\_point), quant\_min,
    // quant\_max)
    Value t0 = rewriter.create<mhlo::DivOp>(loc, op.input(), bcastedScale);
    Value t1 = rewriter.create<mhlo::AddOp>(loc, t0, bcastedZeroPoint);
    Value t2 = rewriter.create<mhlo::ClampOp>(loc, t1, bcastedQuantMin,
                                              bcastedQuantMax);
    Value out = rewriter.create<mhlo::ConvertOp>(loc, op.getType(), t2);
    rewriter.replaceOp(op, out);
    return success();
  }
};

void populateQuantizeAndDequantizePatterns(RewritePatternSet& patterns) {
  // clang-format off
  patterns.insert<
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

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createDiscLowerQuantizeAndDequantizePass() {
  return std::make_unique<DiscLowerQuantizeAndDequantizePass>();
}

}  // namespace disc_ral
}  // namespace mlir

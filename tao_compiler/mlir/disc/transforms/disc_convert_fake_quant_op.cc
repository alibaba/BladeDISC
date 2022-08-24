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

#define DEBUG_TYPE "disc-convert-fake-quant-op"

// This file implements the logic to convert fake_quant annotated graph to
// the real quantized version. An example is shown as following.
//
// convert:
//  const -> fake_quant
//                      \
//  input -> fake_quant -> gemm -> fake_quant ->
// to
//  const -> quantize
//                     \
//  input -> quantize -> quantized_gemm -> dequantize ->

namespace mlir {
namespace disc_ral {
namespace {

struct QuantizedDynamicConvOpConverter
    : public OpRewritePattern<mhlo_disc::FakeQuantOp> {
  using OpRewritePattern<mhlo_disc::FakeQuantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo_disc::FakeQuantOp op,
                                PatternRewriter& rewriter) const override {
    auto convOp = op.input().getDefiningOp<mhlo::DynamicConvOp>();
    if (!convOp) return failure();

    auto inputFakeQuantOp =
        convOp.lhs().getDefiningOp<mhlo_disc::FakeQuantOp>();
    auto weightFakeQuantOp =
        convOp.rhs().getDefiningOp<mhlo_disc::FakeQuantOp>();
    if (!inputFakeQuantOp || !weightFakeQuantOp) return failure();
    if (inputFakeQuantOp.use_signed() != weightFakeQuantOp.use_signed() ||
        inputFakeQuantOp.num_bits() != weightFakeQuantOp.num_bits()) {
      return failure();
    }
    if (inputFakeQuantOp.use_dynamic() != op.use_dynamic() ||
        weightFakeQuantOp.use_dynamic() != op.use_dynamic())
      return failure();

    Location loc = op.getLoc();
    auto buildQuantizedTensorType = [&](mhlo_disc::FakeQuantOp fakeQuantOp) {
      Type quantizedElemType =
          fakeQuantOp.use_signed()
              ? IntegerType::get(rewriter.getContext(), fakeQuantOp.num_bits())
              : IntegerType::get(
                    rewriter.getContext(), fakeQuantOp.num_bits(),
                    mlir::IntegerType::SignednessSemantics::Unsigned);
      auto inputTy = fakeQuantOp.input().getType().cast<RankedTensorType>();
      return RankedTensorType::get(inputTy.getShape(), quantizedElemType,
                                   inputTy.getEncoding());
    };

    // Builds a real quantized op by extracting metadata info from the
    // fake_quant_op
    auto buildQuantizedOpFromFakeQuantOp =
        [&](mhlo_disc::FakeQuantOp fakeQuantOp) {
          auto outTy = buildQuantizedTensorType(fakeQuantOp);
          Value quantizedInput = rewriter.create<mhlo_disc::QuantizeOp>(
              loc, outTy, fakeQuantOp.input(), fakeQuantOp.scale(),
              fakeQuantOp.zero_point(), fakeQuantOp.use_symmetric(),
              fakeQuantOp.axis(), fakeQuantOp.quant_min(),
              fakeQuantOp.quant_max(), fakeQuantOp.use_dynamic());
          return quantizedInput;
        };

    Value quantizedInput = buildQuantizedOpFromFakeQuantOp(inputFakeQuantOp);
    Value quantizedWeight = buildQuantizedOpFromFakeQuantOp(weightFakeQuantOp);
    Value quantizedConv = rewriter.create<mhlo_disc::QuantizedDynamicConvOp>(
        loc, buildQuantizedTensorType(op), quantizedInput, quantizedWeight,
        convOp.d_padding(), inputFakeQuantOp.scale(),
        inputFakeQuantOp.zero_point(), weightFakeQuantOp.scale(),
        weightFakeQuantOp.zero_point(), op.scale(), op.zero_point(),
        *convOp.window_strides(), *convOp.padding(), *convOp.lhs_dilation(),
        *convOp.rhs_dilation(), *convOp.window_reversal(),
        convOp.dimension_numbers(), convOp.feature_group_count(),
        convOp.batch_group_count(), op.use_symmetric(),
        weightFakeQuantOp.axis(), op.use_dynamic());

    Value dequantizedOut = rewriter.create<mhlo_disc::DequantizeOp>(
        loc, op.getType(), quantizedConv, op.scale(), op.zero_point(),
        op.use_symmetric(), op.axis(), op.use_dynamic());

    rewriter.replaceOp(op, dequantizedOut);
    return success();
  }
};

void populateQuantizedPatterns(RewritePatternSet& patterns) {
  // clang-format off
  patterns.insert<
    QuantizedDynamicConvOpConverter
  >(patterns.getContext());
  // clang-format on
}

struct FakeQuantOpToIdentityConverter
    : public OpRewritePattern<mhlo_disc::FakeQuantOp> {
  using OpRewritePattern<mhlo_disc::FakeQuantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo_disc::FakeQuantOp op,
                                PatternRewriter& rewriter) const override {
    rewriter.replaceOp(op, op.input());
    return success();
  }
};

struct DiscConvertFakeQuantOpPass
    : public DiscConvertFakeQuantOpPassBase<DiscConvertFakeQuantOpPass> {
  void runOnOperation() override;
};

void DiscConvertFakeQuantOpPass::runOnOperation() {
  // Setup rewriter patterns.
  MLIRContext& ctx = getContext();
  RewritePatternSet patterns(&ctx);
  populateQuantizedPatterns(patterns);

  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
    return;
  }

  // Remove fake quant ops after quantization conversion.
  RewritePatternSet remove_fake_quant_op_patterns(&ctx);
  remove_fake_quant_op_patterns.insert<FakeQuantOpToIdentityConverter>(&ctx);
  if (failed(applyPatternsAndFoldGreedily(
          getOperation(), std::move(remove_fake_quant_op_patterns)))) {
    signalPassFailure();
    return;
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createDiscConvertFakeQuantOpPass() {
  return std::make_unique<DiscConvertFakeQuantOpPass>();
}

}  // namespace disc_ral
}  // namespace mlir

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

#include "llvm/IR/IntrinsicsAArch64.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "tensorflow/core/util/env_var.h"

#define DEBUG_TYPE "disc-bf16-expansion"

// bf16 has a trivial truncation/extension behavior with F32 that
// can be described in elementary arith operations. Include some
// expansions to efficiently convert including rounding towards
// infinity for f32 to bf16 truncation.

namespace mlir {
namespace disc_ral {
namespace {

bool isBFCVTEnabled() {
  static bool enabled = []() {
    bool enabled = false;
    tensorflow::ReadBoolFromEnvVar("TAO_MLIR_ENABLE_BFCVT", enabled, &enabled);
    return enabled;
  }();
  return enabled;
}
/// Create an integer or index constant.
static Value createConst(Location loc, Type type, int value,
                         PatternRewriter& rewriter) {
  auto attr = rewriter.getIntegerAttr(getElementTypeOrSelf(type), value);
  if (auto shapedTy = dyn_cast<ShapedType>(type)) {
    return rewriter.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(shapedTy, attr));
  }

  return rewriter.create<arith::ConstantOp>(loc, attr);
};

struct BFloat16ExtFOpConverter : public OpRewritePattern<arith::ExtFOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::ExtFOp op,
                                PatternRewriter& rewriter) const final {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto operand = op.getOperand();
    Type operandTy = operand.getType();
    Type resultTy = op.getType();
    Type operandETy = getElementTypeOrSelf(operandTy);
    Type resultETy = getElementTypeOrSelf(resultTy);

    if (!operandETy.isBF16() || !resultETy.isF32()) {
      return failure();
    }

    Type i16Ty = b.getI16Type();
    Type i32Ty = b.getI32Type();
    if (auto shapedTy = dyn_cast<ShapedType>(operandTy)) {
      i16Ty = shapedTy.clone(i16Ty);
      i32Ty = shapedTy.clone(i32Ty);
    }

    Value bitcast = b.create<arith::BitcastOp>(i16Ty, operand);
    Value exti = b.create<arith::ExtUIOp>(i32Ty, bitcast);

    Value c16 = createConst(op.getLoc(), i32Ty, 16, rewriter);
    Value shl = b.create<arith::ShLIOp>(exti, c16);
    Value result = b.create<arith::BitcastOp>(resultTy, shl);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct BFloat16TruncFOpConverter : public OpRewritePattern<arith::TruncFOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::TruncFOp op,
                                PatternRewriter& rewriter) const final {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto operand = op.getOperand();
    Type operandTy = operand.getType();
    Type resultTy = op.getType();
    Type operandETy = getElementTypeOrSelf(operandTy);
    Type resultETy = getElementTypeOrSelf(resultTy);

    if (!operandETy.isF32() || !resultETy.isBF16()) {
      return failure();
    }

#if defined(TAO_AARCH64)
    if (isBFCVTEnabled()) {
      auto intrinsicName =
          StringAttr::get(rewriter.getContext(), "llvm.aarch64.neon.bfcvt");
      SmallVector<Value, 2> args;
      args.push_back(operand);
      rewriter.replaceOpWithNewOp<LLVM::CallIntrinsicOp>(op, resultETy,
                                                         intrinsicName, args);
      return success();
    }
#endif

    Type i1Ty = b.getI1Type();
    Type i16Ty = b.getI16Type();
    Type i32Ty = b.getI32Type();
    Type f32Ty = b.getF32Type();
    if (auto shapedTy = dyn_cast<ShapedType>(operandTy)) {
      i1Ty = shapedTy.clone(i1Ty);
      i16Ty = shapedTy.clone(i16Ty);
      i32Ty = shapedTy.clone(i32Ty);
      f32Ty = shapedTy.clone(f32Ty);
    }

    Value bitcast = b.create<arith::BitcastOp>(i32Ty, operand);
    Value c16 = createConst(op.getLoc(), i32Ty, 16, rewriter);
    Value shr = b.create<arith::ShRUIOp>(bitcast, c16);
    Value trunc = b.create<arith::TruncIOp>(i16Ty, shr);
    Value result = b.create<arith::BitcastOp>(resultTy, trunc);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct DiscBF16ExpansionPass
    : public DiscBF16ExpansionPassBase<DiscBF16ExpansionPass> {
  void runOnOperation() override;
};

void populateExpandBFloat16Patterns(RewritePatternSet& patterns) {
  // clang-format off
  patterns.insert<
    BFloat16ExtFOpConverter,
    BFloat16TruncFOpConverter
  >(patterns.getContext());
  // clang-format on
}

void DiscBF16ExpansionPass::runOnOperation() {
  // Setup rewriter patterns.
  MLIRContext& ctx = getContext();
  RewritePatternSet patterns(&ctx);
  populateExpandBFloat16Patterns(patterns);

  // clang-format on
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
    return;
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createDiscBF16ExpansionPass() {
  return std::make_unique<DiscBF16ExpansionPass>();
}

}  // namespace disc_ral
}  // namespace mlir

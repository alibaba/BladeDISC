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
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"

#define DEBUG_TYPE "disc-algebra-simplifier"

// This file implements some basic optimizations to do algebra simplification.

namespace mlir {
namespace disc_ral {
namespace {

// convert:
//   mhlo.pow(x, const integer n)
// to:
//   %1 = x * 1
//   %2 = x * %1
//   ... ...
//   %n = x * %n-1
struct ExpandPowOp : public OpRewritePattern<mhlo::PowOp> {
  using OpRewritePattern<mhlo::PowOp>::OpRewritePattern;

  LogicalResult expandPowOp(mhlo::PowOp op, PatternRewriter& rewriter,
                            int64_t exponential) const {
    // TODO(disc): support non-positive exponential
    if (exponential <= 0) return failure();
    // TODO(disc): support larger exponential
    if (exponential > 8) return failure();

    Location loc = op.getLoc();
    Value newResult = op->getOperand(0);
    for (int i = 1; i < exponential; ++i)
      newResult =
          rewriter.create<mhlo::MulOp>(loc, newResult, op->getOperand(0));
    rewriter.replaceOp(op, newResult);
    return success();
  }

  LogicalResult extractMultiplerFromConst(mhlo::ConstOp constOp,
                                          int64_t& exponential) const {
    if (!constOp) return failure();

    auto constTy = constOp.getResult().getType().dyn_cast<RankedTensorType>();
    if (!constTy || !constTy.hasStaticShape()) return failure();

    if (!constOp.value().isSplat() && constTy.getNumElements() != 1)
      return failure();

    if (constTy.getElementType().isIntOrIndex()) {
      exponential =
          (*constOp.value().getValues<APInt>().begin()).getSExtValue();
      return success();
    }

    double fpExponential;
    if (constTy.getElementType().isF32()) {
      fpExponential = *constOp.value().getValues<float>().begin();
    } else if (constTy.getElementType().isF64()) {
      fpExponential = *constOp.value().getValues<double>().begin();
    } else {
      // unsupported float types.
      return failure();
    }

    if (static_cast<int64_t>(fpExponential) != fpExponential) return failure();

    exponential = static_cast<int64_t>(fpExponential);
    return success();
  }

  LogicalResult tryToExtractMultipler(mhlo::PowOp op,
                                      int64_t& exponential) const {
    Operation* rhsDefiningOp = op.rhs().getDefiningOp();
    if (!rhsDefiningOp) return failure();

    if (auto constOp = dyn_cast<mhlo::ConstOp>(rhsDefiningOp)) {
      return extractMultiplerFromConst(constOp, exponential);
    }

    // match: scalar const + bcast op pattern
    if (!isa<mhlo::DynamicBroadcastInDimOp, mhlo::BroadcastInDimOp,
             mhlo::BroadcastOp>(rhsDefiningOp)) {
      return failure();
    }
    auto constOp = dyn_cast_or_null<mhlo::ConstOp>(
        rhsDefiningOp->getOperand(0).getDefiningOp());
    return extractMultiplerFromConst(constOp, exponential);
  }

  LogicalResult matchAndRewrite(mhlo::PowOp op,
                                PatternRewriter& rewriter) const override {
    int64_t exponential;
    if (failed(tryToExtractMultipler(op, exponential))) return failure();

    return expandPowOp(op, rewriter, exponential);
  }
};

void populateDiscAlgebraSimplifierPatterns(RewritePatternSet& patterns) {
  // clang-format off
  patterns.insert<
    ExpandPowOp
  >(patterns.getContext());
  // clang-format on
}

struct DiscAlgebraSimplifierPass
    : public DiscAlgebraSimplifierPassBase<DiscAlgebraSimplifierPass> {
  void runOnOperation() override;
};

void DiscAlgebraSimplifierPass::runOnOperation() {
  // Setup rewriter patterns.
  MLIRContext& ctx = getContext();
  RewritePatternSet patterns(&ctx);
  populateDiscAlgebraSimplifierPatterns(patterns);

  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
    return;
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createDiscAlgebraSimplifierPass() {
  return std::make_unique<DiscAlgebraSimplifierPass>();
}

}  // namespace disc_ral
}  // namespace mlir

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
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"
#include "tensorflow/compiler/mlir/disc/transforms/disc_shape_optimization_utils.h"

#define DEBUG_TYPE "disc-algebra-simplifier"

// This file implements some basic optimizations to do algebra simplification.

namespace mlir {
namespace disc_ral {
namespace {

// Returns true if a tensor is constant and all its elements are equal to
// `elemVal`
bool allElementsAreSameValue(Value tensor, int64_t elemVal) {
  auto definingOp = tensor.getDefiningOp();
  if (!definingOp) return false;
  if (isa<mhlo::BroadcastOp, mhlo::BroadcastInDimOp,
          mhlo::DynamicBroadcastInDimOp>(definingOp)) {
    return allElementsAreSameValue(definingOp->getOperand(0), elemVal);
  }
  DenseElementsAttr denseAttr;
  if (!matchPattern(tensor, m_Constant(&denseAttr))) return false;
  if (denseAttr.getNumElements() != 1 && !denseAttr.isSplat()) return false;

  Type elemTy = denseAttr.getElementType();
  if (elemTy.isIntOrIndex()) {
    return (*denseAttr.getValues<APInt>().begin()).getSExtValue() == elemVal;
  } else if (elemTy.isa<FloatType>()) {
    return (*denseAttr.getValues<APFloat>().begin()).convertToDouble() ==
           elemVal;
  }
  return false;
}

// `x + 0` or `0 + x` could be simplified to `x`
struct AddZeroTensorOp : public OpRewritePattern<mhlo::AddOp> {
  using OpRewritePattern<mhlo::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::AddOp op,
                                PatternRewriter& rewriter) const override {
    if (allElementsAreSameValue(op.lhs(), 0)) {
      rewriter.replaceOp(op, op.rhs());
    } else if (allElementsAreSameValue(op.rhs(), 0)) {
      rewriter.replaceOp(op, op.lhs());
    } else {
      return failure();
    }
    return success();
  }
};

// `x * 1` or `1 * x` could be simplified to `x`
struct MulOneTensorOp : public OpRewritePattern<mhlo::MulOp> {
  using OpRewritePattern<mhlo::MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::MulOp op,
                                PatternRewriter& rewriter) const override {
    if (allElementsAreSameValue(op.lhs(), 1)) {
      rewriter.replaceOp(op, op.rhs());
    } else if (allElementsAreSameValue(op.rhs(), 1)) {
      rewriter.replaceOp(op, op.lhs());
    } else {
      return failure();
    }
    return success();
  }
};

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
    newResult.setType(op->getResult(0).getType());
    rewriter.replaceOp(op, newResult);
    return success();
  }

  LogicalResult extractMultiplerFromConst(mhlo::ConstantOp constOp,
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

    if (auto constOp = dyn_cast<mhlo::ConstantOp>(rhsDefiningOp)) {
      return extractMultiplerFromConst(constOp, exponential);
    }

    // match: scalar const + bcast op pattern
    if (!isa<mhlo::DynamicBroadcastInDimOp, mhlo::BroadcastInDimOp,
             mhlo::BroadcastOp>(rhsDefiningOp)) {
      return failure();
    }
    auto constOp = dyn_cast_or_null<mhlo::ConstantOp>(
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

// convert:
//   %1 = mhlo.dynamic_broadcast_in_dim(%0, %target_shape) :
//     (tensor<?x?xf32, [@S0, @S1]>, ...) -> tensor<?x?xf32, [@S0, @S1]>
//   use(%1)
// to:
//   use(%0)
template <typename OpTy>
struct IdentityBroadCastInDimOpCanonicalizationPattern
    : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter& rewriter) const override {
    auto fromTy =
        op->getOperand(0).getType().template dyn_cast<RankedTensorType>();
    auto toTy =
        op->getResult(0).getType().template dyn_cast<RankedTensorType>();
    if (!fromTy || !toTy) return failure();
    // identity bcast op if fromTy & toTy have the same static shape
    if (fromTy.hasStaticShape() && toTy.hasStaticShape() && fromTy == toTy) {
      rewriter.replaceOp(op, op->getOperands());
      return success();
    }

    // (partial) dynamic shape cases.
    // Try to check if the input and out have the same symbolic dim shape.
    auto fromSymbols = getRankedValueSymbolicDimRefs(op->getOperand(0));
    auto toSymbols = getRankedValueSymbolicDimRefs(op->getResult(0));
    if (!fromSymbols || !toSymbols ||
        (*fromSymbols).size() != (*toSymbols).size())
      return failure();

    for (const auto& z : llvm::zip(*fromSymbols, *toSymbols))
      if (std::get<0>(z) != std::get<1>(z)) return failure();

    rewriter.replaceOp(op, op->getOperands());
    return success();
  }
};

// Convert an tf.expand like reshape op followed by a dynamic-broadcast-in-dim
// op to a single dynamic-broadcast-in-dim op. The dim size of expanded dims
// should be one.
//
// An example is like following:
//
//  %0 = ...: tensor<?x?xf32>
//  %d0 = tensor.dim %0, %c0 : tensor<?x?xf32>
//  %d1 = tensor.dim %0, %c1 : tensor<?x?xf32>
//  %new_shape = tensor.from_elements %d0, %d1, %c1 : tensor<3xindex>
//  %1 = "mhlo.dynamic_reshape"(%0, %new_shape) : (tensor<?x?xf32>,
//  tensor<3xindex>) -> tensor<?x?x1xf32>
//  %2 = "mhlo.dynamic_broadcast_in_dim"(%1, %...) {broadcast_dimensions =
//  dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<?x?x1xf32>, tensor<3xindex>)
//  ->tensor<?x?x?xf32>
//
// This pattern will be converted to:
//
//  %2 = "mhlo.dynamic_broadcast_in_dim"(%0, %...) {broadcast_dimensions =
//  dense<[0, 1]> : tensor<3xi64>} : (tensor<?x?xf32>, tensor<3xindex>) ->
//  tensor<?x?x?xf32>
//
// Note that the reshape op can be either `mhlo::DynamicReshapeOp` or
// `mhlo::ReshapeOp`. The expanded dim can be in any axis.
template <typename T>
LogicalResult tryToExtractDimMap(const SmallVectorImpl<T>& in,
                                 const SmallVectorImpl<T>& out,
                                 RankedTensorType outTy,
                                 DenseMap<int64_t, int64_t>& dimMap) {
  int64_t inRank = static_cast<int64_t>(in.size());
  int64_t outRank = static_cast<int64_t>(out.size());

  int64_t outIdx = 0;
  for (int64_t inIdx = 0; inIdx < inRank; ++inIdx, ++outIdx) {
    while (in[inIdx] != out[outIdx]) {
      if (outTy.getShape()[outIdx] != 1) return failure();
      if (++outIdx >= outRank) return failure();
    }
    dimMap[inIdx] = outIdx;
  }
  return success();
}

void newBcastOp(mhlo::BroadcastInDimOp op, PatternRewriter& rewriter,
                SmallVector<Value>& newOperands,
                DenseIntElementsAttr bcastAttr) {
  rewriter.replaceOpWithNewOp<mhlo::BroadcastInDimOp>(
      op, op.getType(), newOperands[0], bcastAttr);
}

void newBcastOp(mhlo::DynamicBroadcastInDimOp op, PatternRewriter& rewriter,
                SmallVector<Value>& newOperands,
                DenseIntElementsAttr bcastAttr) {
  rewriter.replaceOpWithNewOp<mhlo::DynamicBroadcastInDimOp>(
      op, op.getType(), newOperands[0], newOperands[1], bcastAttr);
}

template <typename OpTy>
struct BroadCastInDimOfReshapeOpCanonicalizationPattern
    : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter& rewriter) const override {
    Value bcastOut = op->getResult(0);
    Value reshapeOut = op->getOperand(0);
    auto reshapeOp = reshapeOut.getDefiningOp();
    if (!reshapeOp || !isa<mhlo::ReshapeOp, mhlo::DynamicReshapeOp>(reshapeOp))
      return failure();
    Value reshapeInput = reshapeOp->getOperand(0);

    auto bcastOutTy = bcastOut.getType().template dyn_cast<RankedTensorType>();
    auto reshapeOutTy =
        reshapeOut.getType().template dyn_cast<RankedTensorType>();
    auto reshapeInputTy =
        reshapeInput.getType().template dyn_cast<RankedTensorType>();
    if (!bcastOutTy || !reshapeOutTy || !reshapeInputTy) return failure();

    // early stop if not a tf.expand like reshape
    if (reshapeInputTy.getRank() >= reshapeOutTy.getRank()) return failure();

    bool matched = false;
    // map one dimension of the input of the reshape to one dimension of the
    // output of the reshape.
    DenseMap<int64_t, int64_t> dimMap;

    if (reshapeOutTy.hasStaticShape() && reshapeInputTy.hasStaticShape()) {
      // static shape casse
      if (failed(
              tryToExtractDimMap(llvm::to_vector<4>(reshapeInputTy.getShape()),
                                 llvm::to_vector<4>(reshapeOutTy.getShape()),
                                 reshapeOutTy, dimMap))) {
        return failure();
      }
      matched = true;
    } else {
      // (partial) dynamic shape cases.
      // Try to check if the input and out have the compatible symbolic dim
      // shape.
      auto reshapeInputSymbols = getRankedValueSymbolicDimRefs(reshapeInput);
      auto reshapeOutSymbols = getRankedValueSymbolicDimRefs(reshapeOut);
      if (!reshapeInputSymbols || !reshapeOutSymbols) return failure();

      if (failed(tryToExtractDimMap(*reshapeInputSymbols, *reshapeOutSymbols,
                                    reshapeOutTy, dimMap))) {
        return failure();
      }
      matched = true;
    }

    if (!matched) return failure();

    SmallVector<int64_t> newBcastDims;
    auto oldBcastDims = op.broadcast_dimensions().template getValues<int64_t>();
    for (size_t d = 0, rank = dimMap.size(); d < rank; ++d)
      newBcastDims.push_back(oldBcastDims[dimMap[d]]);

    RankedTensorType ty =
        RankedTensorType::get({static_cast<int64_t>(newBcastDims.size())},
                              rewriter.getIntegerType(64));
    SmallVector<Value> newOperands{reshapeInput};
    for (Value operand : op->getOperands().drop_front())
      newOperands.push_back(operand);

    newBcastOp(op, rewriter, newOperands,
               DenseIntElementsAttr::get(ty, newBcastDims));
    return success();
  }
};

template <typename OpTy>
struct SimplifierFromElementsPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter& rewriter) const override {
    auto loc = op->getLoc();
    Value input = op->getOperand(0);
    Value result = op->getResult(0);
    auto extractOp = input.getDefiningOp();
    if (!isa<tensor::ExtractOp>(extractOp)) {
      return failure();
    }
    auto rankTy = op.getType().template dyn_cast<RankedTensorType>();
    auto extractInput = extractOp->getOperand(0);
    auto outTy = RankedTensorType::get({1}, rankTy.getElementType());
    auto newResult =
        rewriter.create<mhlo::ReshapeOp>(loc, outTy, extractInput).getResult();
    for (auto user : result.getUsers()) {
      user->replaceUsesOfWith(result, newResult);
    }
    return success();
  }
};

void populateDiscAlgebraSimplifierPatterns(RewritePatternSet& patterns) {
  // clang-format off
  patterns.insert<
    BroadCastInDimOfReshapeOpCanonicalizationPattern<mhlo::BroadcastInDimOp>,
    BroadCastInDimOfReshapeOpCanonicalizationPattern<mhlo::DynamicBroadcastInDimOp>,
    ExpandPowOp,
    IdentityBroadCastInDimOpCanonicalizationPattern<mhlo::BroadcastInDimOp>,
    IdentityBroadCastInDimOpCanonicalizationPattern<mhlo::BroadcastOp>,
    IdentityBroadCastInDimOpCanonicalizationPattern<mhlo::DynamicBroadcastInDimOp>,
    SimplifierFromElementsPattern<tensor::FromElementsOp>
  >(patterns.getContext());

  // zero tensor related patterns
  patterns.insert<
    AddZeroTensorOp,
    MulOneTensorOp
  >(patterns.getContext());
  // clang-format on
}

struct DiscAlgebraSimplifierPass
    : public DiscAlgebraSimplifierPassBase<DiscAlgebraSimplifierPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mhlo::MhloDialect>();
  }
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

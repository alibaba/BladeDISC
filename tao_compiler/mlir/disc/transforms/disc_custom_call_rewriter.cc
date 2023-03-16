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

#include "llvm/Support/Debug.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/disc/IR/hlo_disc_ops.h"
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/PassDetail.h"

#define DEBUG_TYPE "disc-custom-call-rewriter"

// This file implements the logic to rewrite mhlo_disc::CustomCallV2Op
// according to its layout attributes.

namespace mlir {
namespace disc_ral {
namespace {

LogicalResult inferTypeAndPermutation(Value val, StringRef layout,
                                      StringRef expectedLayout, Type& newType,
                                      SmallVector<int64_t>& permutation) {
  int mapped = 0;
  auto type = val.getType().cast<RankedTensorType>();
  auto newShape = llvm::to_vector<4>(type.getShape());
  permutation = SmallVector<int64_t>(type.getRank(), -1);
  for (size_t i = 0; i < layout.size(); ++i) {
    auto offset = expectedLayout.find(layout[i]);
    if (offset == std::string::npos) return failure();
    if (permutation[offset] == -1) ++mapped;
    permutation[offset] = i;
    newShape[offset] = type.getShape()[i];
  }
  if (mapped != layout.size()) return failure();
  newType = RankedTensorType::get(newShape, type.getElementType(),
                                  type.getEncoding());
  return success();
}

LogicalResult rewriteValue(PatternRewriter& rewriter, Location loc, Value val,
                           StringRef layout, StringRef expectedLayout,
                           SmallVector<Value>& newOperands) {
  if (layout == expectedLayout) {
    newOperands.push_back(val);
    return success();
  }

  Type newType;
  SmallVector<int64_t> permutation;
  if (failed(inferTypeAndPermutation(val, layout, expectedLayout, newType,
                                     permutation)))
    return failure();

  auto permutationAttr = GetI64ElementsAttr(permutation, &rewriter);
  newOperands.push_back(
      rewriter.create<mhlo::TransposeOp>(loc, newType, val, permutationAttr));
  return success();
}

struct CustomCallV2LayoutRewriter
    : public OpRewritePattern<mhlo_disc::CustomCallV2Op> {
  using OpRewritePattern<mhlo_disc::CustomCallV2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo_disc::CustomCallV2Op op,
                                PatternRewriter& rewriter) const override {
    // All ins and ous match: no need to rewrite.
    if (op.getInputLayouts() == op.getExpectedInputLayouts() &&
        op.getOutputLayouts() == op.getExpectedOutputLayouts()) {
      return failure();
    }

    SmallVector<Value> newOperands;
    for (const auto& z : llvm::zip(op->getOperands(), op.parseInputLayouts(),
                                   op.parseExpectedInputLayouts())) {
      if (failed(rewriteValue(rewriter, op.getLoc(), std::get<0>(z),
                              std::get<1>(z), std::get<2>(z), newOperands)))
        return op->emitOpError()
               << "failed to rewrite the operand of custom call op\n";
    }

    SmallVector<Type> newResultTypes;
    for (const auto& z : llvm::zip(op->getResults(), op.parseOutputLayouts(),
                                   op.parseExpectedOutputLayouts())) {
      Type newType;
      SmallVector<int64_t> permutation;
      if (failed(inferTypeAndPermutation(std::get<0>(z), std::get<1>(z),
                                         std::get<2>(z), newType, permutation)))
        return op->emitOpError()
               << "failed to infer new result type of custom call op\n";
      newResultTypes.push_back(newType);
    }

    auto newCustomOp = rewriter.create<mhlo_disc::CustomCallV2Op>(
        op.getLoc(), newResultTypes, newOperands, op.getCallTargetName(),
        op.getCustomAttrs(), op.getHasSideEffect(), op.getDevice(),
        op.getInputPlacements(), op.getOutputPlacements(),
        op.getExpectedInputLayouts(), op.getExpectedOutputLayouts(),
        op.getExpectedInputLayouts(), op.getExpectedOutputLayouts());

    SmallVector<Value> newResults;
    for (const auto& z :
         llvm::zip(newCustomOp->getResults(), op.parseOutputLayouts(),
                   op.parseExpectedOutputLayouts())) {
      if (failed(rewriteValue(rewriter, op.getLoc(), std::get<0>(z),
                              std::get<2>(z), std::get<1>(z), newResults)))
        return op->emitOpError()
               << "failed to rewrite the result of custom call op\n";
    }

    rewriter.replaceOp(op, newResults);
    return success();
  }
};

struct DiscCustomCallRewriterPass
    : public DiscCustomCallRewriterPassBase<DiscCustomCallRewriterPass> {
  void runOnOperation() override;
};

void DiscCustomCallRewriterPass::runOnOperation() {
  // Setup rewriter patterns.
  MLIRContext& ctx = getContext();
  RewritePatternSet patterns(&ctx);
  patterns.insert<CustomCallV2LayoutRewriter>(&ctx);

  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
    return;
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createDiscCustomCallRewriterPass() {
  return std::make_unique<DiscCustomCallRewriterPass>();
}

}  // namespace disc_ral
}  // namespace mlir

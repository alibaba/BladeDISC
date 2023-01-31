/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// This file implements the logic to flattern memref to 1D format.

#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/disc/transforms/PassDetail.h"

namespace mlir {
namespace disc_ral {

namespace {

using memref::ReinterpretCastOp;

struct ReinterpretCastOpConverter : public OpRewritePattern<ReinterpretCastOp> {
 public:
  using OpRewritePattern<ReinterpretCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReinterpretCastOp op,
                                PatternRewriter& rewriter) const override;
};

LogicalResult ReinterpretCastOpConverter::matchAndRewrite(
    ReinterpretCastOp op, PatternRewriter& rewriter) const {
  Location loc = op.getLoc();
  auto alloc =
      dyn_cast_or_null<memref::AllocOp>(op.getSource().getDefiningOp());
  if (!alloc) return failure();

  auto allocTy = alloc.getResult().getType().cast<MemRefType>();
  if (allocTy != op.getType()) return failure();

  if (!allocTy.hasStaticShape()) {
    alloc->setOperands(op.sizes());
  }
  StringRef attrName = disc_shape::SymbolicDimOp::getSymbolicDimAttrName();
  if (op->hasAttr(attrName)) {
    alloc->setAttr(attrName, op->getAttr(attrName));
  }
  rewriter.replaceOp(op, alloc->getResults());
  return success();
}

struct DiscMemrefCanonicalizer
    : DiscMemrefCanonicalizerBase<DiscMemrefCanonicalizer> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext* ctx = func.getContext();

    // Populate patterns.
    RewritePatternSet patterns(ctx);
    patterns.insert<ReinterpretCastOpConverter>(ctx);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createDiscMemrefCanonicalizerPass() {
  return std::make_unique<DiscMemrefCanonicalizer>();
}

}  // namespace disc_ral
}  // namespace mlir

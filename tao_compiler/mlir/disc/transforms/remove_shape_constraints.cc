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

#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/rewriters.h"

namespace mlir {
namespace disc_ral {

using mhlo::CstrReshapableOp;

namespace {

class CstrReshapableOpConversion : public OpRewritePattern<CstrReshapableOp> {
 public:
  using OpRewritePattern<CstrReshapableOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CstrReshapableOp op,
                                PatternRewriter& rewriter) const override;
};

LogicalResult CstrReshapableOpConversion::matchAndRewrite(
    CstrReshapableOp op, PatternRewriter& rewriter) const {
  rewriter.replaceOpWithNewOp<shape::ConstWitnessOp>(op.getOperation(), true);
  return success();
}

class RemoveShapeConstraintsPass
    : public RemoveShapeConstraintsPassBase<RemoveShapeConstraintsPass> {
  void runOnOperation() override;
};

void RemoveShapeConstraintsPass::runOnOperation() {
  // Setup target legality.
  MLIRContext& ctx = getContext();

  // Setup conversion patterns.
  RewritePatternSet patterns(&ctx);
  // clang-format: off
  patterns.insert<CstrReshapableOpConversion>(&ctx);
  // clang-format: on
  populateRemoveShapeConstraintsPatterns(patterns);

  func::FuncOp func = getOperation();
  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
    func.emitError("applyPatternsAndFoldGreedily does not converge");
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createDiscRemoveShapeConstraintsPass() {
  return std::make_unique<RemoveShapeConstraintsPass>();
}

}  // namespace disc_ral
}  // namespace mlir

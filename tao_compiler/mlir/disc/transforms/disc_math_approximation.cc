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

// This file implements approximation logic for some special math ops.

#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/disc/transforms/PassDetail.h"

namespace mlir {
namespace disc_ral {

namespace {

struct DiscMathApproximationPass
    : DiscMathApproximationPassBase<DiscMathApproximationPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext* ctx = func.getContext();

    // Populate patterns.
    RewritePatternSet patterns(ctx);
    populateMathPolynomialApproximationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createDiscMathApproximationPass() {
  return std::make_unique<DiscMathApproximationPass>();
}

}  // namespace disc_ral
}  // namespace mlir

// Copyright 2021 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir/Dialect/MemRef/IR/MemRef.h"               // TF:llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"             // TF:llvm-project
#include "mlir/IR/Location.h"                            // TF:llvm-project
#include "mlir/IR/MLIRContext.h"                         // TF:llvm-project
#include "mlir/IR/PatternMatch.h"                        // TF:llvm-project
#include "mlir/Pass/Pass.h"                              // TF:llvm-project
#include "mlir/Transforms/DialectConversion.h"           // TF:llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // TF:llvm-project
#include "mlir/Transforms/Passes.h"                      // TF:llvm-project
#include "transforms/PassDetail.h"

namespace mlir {
namespace disc_ral {

namespace {

/// Converts `atomic_rmw` that cannot be lowered to a simple atomic op with
/// memref::AtomicRMWOpLowering pattern, e.g. with "mulf" attributes, to
/// `generic_atomic_rmw` with the expanded code. This is a supplement to
/// `StdExpandOpsPass`.
///
/// After lowering, the IR looks like:
///
/// %x = std.generic_atomic_rmw %F[%i] : memref<10xf32> {
/// ^bb0(%current: f32):
///   %new_value = ...
///   atomic_yield %new_value : f32
/// }
struct UnhandledAtomicRMWConverter
    : public OpRewritePattern<memref::AtomicRMWOp> {
  using OpRewritePattern<memref::AtomicRMWOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AtomicRMWOp op,
                                PatternRewriter& rewriter) const override {
    // Currently, we only deal with atomic mulf operation. More operations can
    // be supported easily here.
    if (op.kind() != arith::AtomicRMWKind::mulf) {
      return failure();
    }

    Location loc = op.getLoc();
    memref::GenericAtomicRMWOp genericOp =
        rewriter.create<memref::GenericAtomicRMWOp>(loc, op.memref(),
                                                    op.indices());
    OpBuilder bodyBuilder =
        OpBuilder::atBlockEnd(genericOp.getBody(), rewriter.getListener());

    Value lhs = genericOp.getCurrentValue();
    Value rhs = op.value();
    Value reductionOp = getReductionOp(op.kind(), bodyBuilder, loc, lhs, rhs);
    bodyBuilder.create<memref::AtomicYieldOp>(loc, reductionOp);

    rewriter.replaceOp(op, genericOp.getResult());
    return success();
  }
};

struct UnhandledAtomicRMWConverterPass
    : public UnhandledAtomicRMWConverterPassBase<
          UnhandledAtomicRMWConverterPass> {
  void runOnOperation() override {
    MLIRContext& ctx = getContext();

    RewritePatternSet patterns(&ctx);
    patterns.add<UnhandledAtomicRMWConverter>(&ctx);

    if (failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>>
createDiscUnhandledAtomicRMWConverterPass() {
  return std::make_unique<UnhandledAtomicRMWConverterPass>();
}

}  // namespace disc_ral
}  // namespace mlir

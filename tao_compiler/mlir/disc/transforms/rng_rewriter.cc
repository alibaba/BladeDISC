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

#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"       // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"                // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"               // from @llvm-project
#include "mlir/IR/Attributes.h"                          // from @llvm-project
#include "mlir/IR/Location.h"                            // from @llvm-project
#include "mlir/IR/MLIRContext.h"                         // from @llvm-project
#include "mlir/IR/Operation.h"                           // from @llvm-project
#include "mlir/IR/PatternMatch.h"                        // from @llvm-project
#include "mlir/Pass/Pass.h"                              // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"                      // from @llvm-project
#include "tensorflow/compiler/mlir/disc/IR/hlo_disc_ops.h"
#include "tensorflow/compiler/mlir/disc/IR/rng_uniform_custom_call_op.h"
#include "tensorflow/compiler/mlir/disc/disc_util.h"
#include "transforms/PassDetail.h"

using llvm::StringRef;
using std::string;

namespace mlir {

namespace disc_ral {

StringAttr PackRandomUniformBackendConfig(IntegerAttr seed, IntegerAttr seed2,
                                          PatternRewriter* rewriter) {
  mhlo_disc::RngUniformBackendConfig config(seed.getValue().getSExtValue(),
                                            seed2.getValue().getSExtValue());
  std::string str;
  llvm::raw_string_ostream ostream(str);
  ostream << ::llvm::json::toJSON(config);
  return rewriter->getStringAttr(ostream.str());
}

namespace {
struct RngConvert : public OpRewritePattern<mhlo::RngOp> {
  explicit RngConvert(MLIRContext* context) : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(mhlo::RngOp op,
                                PatternRewriter& rewriter) const override {
    if (op.rng_distribution() != ::mlir::mhlo::RngDistribution::UNIFORM) {
      return failure();
    }

    auto cfg = PackRandomUniformBackendConfig(
        rewriter.getIntegerAttr(rewriter.getI32Type(), 1),
        rewriter.getIntegerAttr(rewriter.getI32Type(), 2), &rewriter);

    auto custom_call_op = rewriter.create<mlir::mhlo_disc::CustomCallOp>(
        op.getLoc(), TypeRange{op.getType()},
        ValueRange{op.a(), op.b(), op.shape()},
        rewriter.getStringAttr("rng_uniform"), rewriter.getBoolAttr(false),
        cfg);
    rewriter.replaceOp(op, {custom_call_op.getResult(0)});
    return success();
  }
};

struct RngRewriterPass : public RngRewriterPassBase<RngRewriterPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    // TODO: if needs to do const reformat, we need the xla_hlo.dot with its
    // inputs

    MLIRContext* ctx = func.getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<RngConvert>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      func.emitError("applyPatternsAndFoldGreedily does not converge");
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createDiscRngRewriterPass() {
  return std::make_unique<RngRewriterPass>();
}

}  // namespace disc_ral
}  // namespace mlir

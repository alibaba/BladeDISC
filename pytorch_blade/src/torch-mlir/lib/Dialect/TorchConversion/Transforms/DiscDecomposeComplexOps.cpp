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

#include <mlir-hlo/Dialect/mhlo/IR/chlo_ops.h> // from tf repo

#include "torch-mlir/Conversion/MhloPasses.h"
#include "torch-mlir/Conversion/TorchToMhlo/TorchToMhlo.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::TorchConversion;

namespace {
class DecomposeAtenHardtanhOp : public OpRewritePattern<AtenHardtanhOp> {
 public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenHardtanhOp op, PatternRewriter& rewriter)
      const override {
    Location loc = op.getLoc();
    Value input = op.self();
    BaseTensorType inputType = input.getType().cast<BaseTensorType>();

    // result = min(maxVal, max(minVal, x))
    /**
    Value minVal = chlo::getConstantLike(rewriter, loc, op.min_val(), input);
    Value maxVal = chlo::getConstantLike(rewriter, loc, op.max_val(), input);
    Value maxResult =
        rewriter.create<AtenMaximumOp>(loc, inputType, input, minVal);
    rewriter.replaceOpWithNewOp<AtenMinimumOp>(op, op.getType(), maxVal,
                                               maxResult);
    **/
    return success();
  }
};
} // namespace

namespace {

class DiscDecomposeComplexOpsPass
    : public DiscDecomposeComplexOpsBase<DiscDecomposeComplexOpsPass> {
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalDialect<Torch::TorchDialect>();

    patterns.add<DecomposeAtenHardtanhOp>(context);
    target.addIllegalOp<AtenHardtanhOp>();
  }
};

} //  namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::torch::TorchConversion::
    createDiscDecomposeComplexOpsPass() {
  return std::make_unique<DiscDecomposeComplexOpsPass>();
}
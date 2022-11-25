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

#include "torch-mlir/Conversion/MhloPasses.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

#include <iostream>
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/ChloOps.h"
using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::TorchConversion;

namespace {
template <typename AtenOpT>
class UpgradeAtenOp : public OpRewritePattern<AtenOpT> {
 public:
  using OpRewritePattern<AtenOpT>::OpRewritePattern;
  LogicalResult matchAndRewrite(AtenOpT op, PatternRewriter& rewriter)
      const override;
};
} // namespace

namespace {
template <>
LogicalResult UpgradeAtenOp<AtenGeluOp>::matchAndRewrite(
    AtenGeluOp op,
    PatternRewriter& rewriter) const {
  if (op.getNumOperands() == 1) {
    llvm::dbgs() << "Upgrading legacy op: " << op << "\n";
    Value cstStrNone =
        rewriter.create<Torch::ConstantStrOp>(op.getLoc(), "none");
    rewriter.replaceOpWithNewOp<AtenGeluOp>(
        op, op.getType(), op.self(), cstStrNone);
    return success();
  }
  return failure();
}
} // namespace

namespace {
class DiscUpgradeLegacyOpsPass
    : public DiscUpgradeLegacyOpsBase<DiscUpgradeLegacyOpsPass> {
  void runOnOperation() override {
    MLIRContext* context = &getContext();

    ConversionTarget target(*context);
    target.addLegalDialect<Torch::TorchDialect>();

    RewritePatternSet patterns(context);
    auto opIsDynamicallyLegal = [&](Operation* op) {
      return not failed(mlir::verify(op));
    };

#define UPGRADE_ATENOP_PATTERN(AtenOp)                        \
  target.addDynamicallyLegalOp<AtenOp>(opIsDynamicallyLegal); \
  patterns.add<UpgradeAtenOp<AtenOp>>(context)
    UPGRADE_ATENOP_PATTERN(AtenGeluOp);
#undef UPGRADE_ATENOP_PATTERN

    if (failed(applyPartialConversion(
            getOperation(), target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} //  namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::torch::TorchConversion::
    createDiscUpgradeLegacyOpsPass() {
  return std::make_unique<DiscUpgradeLegacyOpsPass>();
}

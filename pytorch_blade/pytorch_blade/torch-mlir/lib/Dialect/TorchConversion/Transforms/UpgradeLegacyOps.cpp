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

#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/DialectConversion.h"

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

template <typename AtenOpT>
bool isLegalAtenOp(AtenOpT op) {
  return true;
}
} // namespace

// Upgrade AtenGeluOp
namespace {
template <>
bool isLegalAtenOp<AtenGeluOp>(AtenGeluOp op) {
  // before 1.12: gelu(Tensor self) -> Tensor
  // since  1.12: gelu(Tensor self, *, str approximate='none') -> Tensor
  return op.getNumOperands() == 2;
}

template <>
LogicalResult UpgradeAtenOp<AtenGeluOp>::matchAndRewrite(
    AtenGeluOp op,
    PatternRewriter& rewriter) const {
  if (op.getNumOperands() == 1) {
    llvm::dbgs() << "Upgrading legacy op: " << op << "\n";
    Value cstStrNone =
        rewriter.create<Torch::ConstantStrOp>(op.getLoc(), "none");
    rewriter.replaceOpWithNewOp<AtenGeluOp>(
        op, op.getType(), op.getSelf(), cstStrNone);
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

#define UPGRADE_ATENOP_PATTERN(AtenOp)                          \
  target.addDynamicallyLegalOp<AtenOp>(&isLegalAtenOp<AtenOp>); \
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

#undef TORCH_VERSION_LT

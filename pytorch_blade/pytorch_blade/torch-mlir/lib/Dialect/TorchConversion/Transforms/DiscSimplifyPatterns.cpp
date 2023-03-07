// Copyright 2023 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir/Transforms/DialectConversion.h"

#include "torch-mlir/Conversion/MhloPasses.h"
#include "torch-mlir/Conversion/TorchToMhlo/TorchToMhlo.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::TorchConversion;
namespace {

class SimplifyGetitemOp : public OpConversionPattern<Aten__Getitem__TOp> {
 public:
  using OpConversionPattern<Aten__Getitem__TOp>::OpConversionPattern;
  using OpAdaptor = typename Aten__Getitem__TOp::Adaptor;
  LogicalResult matchAndRewrite(
      Aten__Getitem__TOp op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto input = op.getOperand(0);
    auto index = op.getOperand(1);
    auto result = op.getResult();

    int64_t indexInt;
    if (!matchPattern(op.getIdx(), m_TorchConstantInt(&indexInt)))
      return failure();

    auto listOp = input.getDefiningOp<PrimListConstructOp>();
    if (!listOp)
      return failure();

    SmallVector<Value> torchTensors;
    if (!getListConstructElements(listOp, torchTensors)) {
      return failure();
    }
    for (Operation* user : result.getUsers()) {
      user->replaceUsesOfWith(result, torchTensors[indexInt]);
    }
    op.erase();
    return success();
  }
};

class DiscSimplifyPatternsPass
    : public DiscSimplifyPatternsPassBase<DiscSimplifyPatternsPass> {
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<Torch::TorchDialect>();

    RewritePatternSet patterns(context);
    patterns.add<SimplifyGetitemOp>(context);

    if (failed(applyPartialConversion(
            getOperation(), target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::torch::TorchConversion::
    createDiscSimplifyPatternsPass() {
  return std::make_unique<DiscSimplifyPatternsPass>();
}

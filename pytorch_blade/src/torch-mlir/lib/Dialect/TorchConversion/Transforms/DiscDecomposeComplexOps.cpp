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
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

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
template <typename AtenOpT>
class ConvertAtenOp : public OpConversionPattern<AtenOpT> {
 public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult matchAndRewrite(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override;
};

template <>
LogicalResult ConvertAtenOp<AtenHardtanhOp>::matchAndRewrite(
    AtenHardtanhOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  Location loc = op.getLoc();
  Value input = adaptor.self();
  BaseTensorType inputType = input.getType().cast<BaseTensorType>();

  auto sizeListType =
      Torch::ListType::get(Torch::IntType::get(op.getContext()));
  Value sizeList =
      rewriter.create<AtenSizeOp>(op.getLoc(), sizeListType, input);

  SmallVector<int64_t> empty;
  Type tensorType = inputType.getWithSizesAndDtype(
      llvm::makeArrayRef(empty), rewriter.getF32Type());

  Value minTensor =
      rewriter.create<PrimNumToTensorScalarOp>(loc, tensorType, op.min_val());
  Value minValue = rewriter.create<AtenBroadcastToOp>(
      loc, op.getType(), minTensor, sizeList);
  Value maxResult =
      rewriter.create<AtenMaximumOp>(loc, inputType, input, minValue);

  Value maxTensor =
      rewriter.create<PrimNumToTensorScalarOp>(loc, tensorType, op.max_val());
  Value maxValue = rewriter.create<AtenBroadcastToOp>(
      loc, op.getType(), maxTensor, sizeList);
  rewriter.replaceOpWithNewOp<AtenMinimumOp>(
      op, op.getType(), maxResult, maxValue);

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenNativeDropoutOp>::matchAndRewrite(
    AtenNativeDropoutOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  Location loc = op.getLoc();
  Value input = op.input();
  Value prob = op.p();
  bool train = false;
  if (!matchPattern(op.train(), m_TorchConstantBool(&train)))
    return rewriter.notifyMatchFailure(op, "train must be a boolean constant");

  BaseTensorType inputType = input.getType().cast<BaseTensorType>();
  if (!train) {
    // TODO(yancey.yx): supports inference mode
    return op.emitError(
        "native_dropout does not support argument train is false");
  }
  if (!inputType.hasDtype() || !inputType.getDtype().isa<mlir::FloatType>())
    return rewriter.notifyMatchFailure(
        op, "only support floating type input for training mode");
  Value noneVal = rewriter.create<ConstantNoneOp>(loc);
  Value floatOne =
      rewriter.create<ConstantFloatOp>(loc, rewriter.getF64FloatAttr(1.0));
  Value oneMinusP = rewriter.create<AtenSubFloatOp>(loc, floatOne, prob);
  Value boolMask = rewriter.create<ValsemVariantAtenBernoulliFloatOp>(
      loc, inputType, input, oneMinusP, /*generator=*/noneVal);
  Value maskedInput =
      rewriter.create<AtenMulTensorOp>(loc, inputType, boolMask, input);
  Value output =
      rewriter.create<AtenMulScalarOp>(loc, inputType, maskedInput, oneMinusP);
  Value one =
      rewriter.create<Torch::ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
  boolMask = rewriter.create<AtenGeScalarOp>(
      loc, op.getResult(1).getType(), boolMask, one);
  rewriter.replaceOp(op, {output, boolMask});
  return success();
}

} // namespace

namespace {

class DiscDecomposeComplexOpsPass
    : public DiscDecomposeComplexOpsBase<DiscDecomposeComplexOpsPass> {
  void runOnOperation() override {
    MLIRContext* context = &getContext();

    ConversionTarget target(*context);
    target.addLegalDialect<Torch::TorchDialect>();

    RewritePatternSet patterns(context);
    patterns.add<ConvertAtenOp<AtenHardtanhOp>>(context);
    target.addIllegalOp<AtenHardtanhOp>();

    patterns.add<ConvertAtenOp<AtenNativeDropoutOp>>(context);
    target.addIllegalOp<AtenNativeDropoutOp>();

    if (failed(applyPartialConversion(
            getOperation(), target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} //  namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::torch::TorchConversion::
    createDiscDecomposeComplexOpsPass() {
  return std::make_unique<DiscDecomposeComplexOpsPass>();
}
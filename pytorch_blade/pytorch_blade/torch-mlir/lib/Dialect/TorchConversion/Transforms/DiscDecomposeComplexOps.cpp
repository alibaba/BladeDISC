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
#include "torch-mlir/Conversion/TorchToMhlo/TorchToMhlo.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/ChloOps.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::TorchConversion;

namespace {
// Helper to create a tensor filled with the given scalar. Scalar would be
// converted the to the element type of the given tensor type.
static Value createInitTensor(
    PatternRewriter& rewriter,
    Location loc,
    Type resultType,
    Value scalar,
    Value sizeList) {
  Value noneVal = rewriter.create<ConstantNoneOp>(loc);
  return rewriter.create<AtenFullOp>(
      loc,
      resultType,
      sizeList,
      scalar,
      /*dtype=*/noneVal,
      /*layout=*/noneVal,
      /*device=*/noneVal,
      /*memory_format=*/noneVal);
}

// Helper to create a rank 0 tensor filled with the given `scalar`. `scalar`
// would be converted to the element type of the given `inputType`.
static Value createRank0Tensor(
    PatternRewriter& rewriter,
    Location loc,
    BaseTensorType inputType,
    Value scalar) {
  SmallVector<int64_t> sizes;
  Type rank0TensorTy = inputType.getWithSizesAndDtype(
      makeArrayRef(sizes), inputType.getOptionalDtype());
  Value dimList = rewriter.create<PrimListConstructOp>(
      loc,
      Torch::ListType::get(Torch::IntType::get(inputType.getContext())),
      ValueRange{});
  return createInitTensor(rewriter, loc, rank0TensorTy, scalar, dimList);
}

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
LogicalResult ConvertAtenOp<OperatorOp>::matchAndRewrite(
    OperatorOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  Location loc = op.getLoc();
  auto name = op.name();
  if (std::string("aten._autocast_to_reduced_precision") == name ||
      std::string("aten._autocast_to_full_precision") == name) {
    // dtype has been infered in PropagateInputShapes pass
    auto inTy = op.getOperand(0).getType();
    auto outTy = op.getResult(0).getType();
    auto inDtype = outTy.template dyn_cast<BaseTensorType>().getDtype();
    auto outDtype = outTy.template dyn_cast<BaseTensorType>().getDtype();
    if (inDtype != outDtype) {
      auto dtype = getDtypeIntValueForType(rewriter, loc, outDtype);
      Value constantNone = rewriter.create<ConstantNoneOp>(loc);
      Value constantTrue = rewriter.create<ConstantBoolOp>(loc, true);
      rewriter.replaceOpWithNewOp<AtenToDtypeOp>(
          op,
          outTy,
          op.getOperand(0),
          dtype,
          /*non-blocking*/ constantTrue,
          /*copy*/ constantTrue,
          /*memomry format*/ constantNone);

    } else {
      rewriter.replaceOp(op, {op.getOperand(0)});
    }
    return success();
  }
  return failure();
}

template <>
LogicalResult ConvertAtenOp<AtenBatchNormOp>::matchAndRewrite(
    AtenBatchNormOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  Location loc = op.getLoc();
  MLIRContext* context = op.getContext();
  Value input = op.input();
  Value weight = op.weight();
  Value bias = op.bias();
  Value runningMean = op.running_mean();
  Value runningVar = op.running_var();
  Value training = op.training();
  Value momentum = op.momentum();
  Value eps = op.eps();

  auto outTy = op.getType().dyn_cast<BaseTensorType>();
  auto meanVarTy =
      outTy.getWithSizesAndDtype({outTy.getSizes()[1]}, outTy.getDtype());
  auto nativeBatchNorm = rewriter.create<AtenNativeBatchNormOp>(
      op.getLoc(),
      outTy,
      meanVarTy,
      meanVarTy,
      input,
      weight,
      bias,
      runningMean,
      runningVar,
      training,
      momentum,
      eps);
  rewriter.replaceOp(op, nativeBatchNorm.getResult(0));
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenMaskedFillScalarOp>::matchAndRewrite(
    AtenMaskedFillScalarOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  rewriter.replaceOpWithNewOp<AtenWhereScalarSelfOp>(
      op, op.getType(), op.mask(), op.value(), op.self());
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenMaskedFillTensorOp>::matchAndRewrite(
    AtenMaskedFillTensorOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  rewriter.replaceOpWithNewOp<AtenWhereSelfOp>(
      op, op.getType(), op.mask(), op.value(), op.self());
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenPowTensorScalarOp>::matchAndRewrite(
    AtenPowTensorScalarOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  Location loc = op.getLoc();
  auto resType = op.getType().cast<BaseTensorType>();
  Value expTensor = createRank0Tensor(rewriter, loc, resType, op.exponent());

  rewriter.replaceOpWithNewOp<AtenPowTensorTensorOp>(
      op, op.getType(), op.self(), expTensor);
  return success();
}

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
template <>
LogicalResult ConvertAtenOp<AtenNllLossForwardOp>::matchAndRewrite(
    AtenNllLossForwardOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  Location loc = op.getLoc();
  Value self = op.self();
  Value target = op.target();
  Value weight = op.weight();
  int64_t reduction;
  if (!matchPattern(op.reduction(), m_TorchConstantInt(&reduction)))
    return rewriter.notifyMatchFailure(op, "reduction must be a constant");

  int64_t ignoreIndex;
  if (!matchPattern(op.ignore_index(), m_TorchConstantInt(&ignoreIndex)))
    return rewriter.notifyMatchFailure(
        op, "unimplemented, the ignore_index operand is not -100");

  Value constantNone = rewriter.create<ConstantNoneOp>(loc);
  Value zero =
      rewriter.create<Torch::ConstantIntOp>(loc, rewriter.getI64IntegerAttr(0));
  Value one =
      rewriter.create<Torch::ConstantIntOp>(loc, rewriter.getI64IntegerAttr(1));
  Value batch = rewriter.create<AtenSizeIntOp>(loc, self, zero);

  Value totalWeight;
  // TODO: Incorporate the weight argument.
  if (weight.getType().isa<Torch::NoneType>()) {
    totalWeight = rewriter.create<AtenFloatScalarOp>(loc, batch);
    totalWeight = rewriter.create<PrimNumToTensorScalarOp>(
        loc, op->getResult(1).getType(), totalWeight);
  } else {
    return rewriter.notifyMatchFailure(
        op, "unimplemented, the weight operand is not incorporated.");
  }

  Value result = rewriter.create<AtenIndexSelectOp>(
      loc, self.getType(), self, one, target);
  if (reduction == torch_upstream::Reduction::None) {
    rewriter.replaceOp(op, {result, totalWeight});
    return success();
  }

  auto resultType = op->getResult(0).getType();
  result = rewriter.create<AtenSumOp>(loc, resultType, result, constantNone)
               .getResult();
  if (reduction == torch_upstream::Reduction::Mean) {
    result = rewriter.create<AtenDivTensorOp>(
        loc, result.getType(), result, totalWeight);
  }
  rewriter.replaceOp(op, {result, totalWeight});
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
    auto opIsDynamicallyLegal = [&](OperatorOp op) {
      if (std::string("aten._autocast_to_reduced_precision") == op.name() ||
          std::string("aten._autocast_to_full_precision") == op.name()) {
        return false;
      }
      return true;
    };

    // Won't mark OperatorOp as illegal, some custom operator may remain
    // unconverted.
    target.addDynamicallyLegalOp<OperatorOp>(opIsDynamicallyLegal);
    patterns.add<ConvertAtenOp<OperatorOp>>(context);

#define INSERT_ATENOP_PATTERN(AtenOp) \
  target.addIllegalOp<AtenOp>();      \
  patterns.add<ConvertAtenOp<AtenOp>>(context)

    INSERT_ATENOP_PATTERN(AtenBatchNormOp);
    INSERT_ATENOP_PATTERN(AtenHardtanhOp);
    INSERT_ATENOP_PATTERN(AtenMaskedFillScalarOp);
    INSERT_ATENOP_PATTERN(AtenMaskedFillTensorOp);
    INSERT_ATENOP_PATTERN(AtenNativeDropoutOp);
    INSERT_ATENOP_PATTERN(AtenNllLossForwardOp);
    INSERT_ATENOP_PATTERN(AtenPowTensorScalarOp);

#undef INSERT_ATENOP_PATTERN

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

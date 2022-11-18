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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/ChloOps.h"
#include "tensorflow/compiler/mlir/disc/IR/hlo_disc_ops.h"
#include "torch-mlir/Conversion/MhloPasses.h"
#include "torch-mlir/Conversion/TorchToMhlo/TorchToMhlo.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"
using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::TorchConversion;

namespace {

static std::vector<std::string> quantizationOpList{
    "torch_blade.fake_quant",
    "torch_blade.quantize",
    "torch_blade.dequantize"};
static std::string customCallName = "torch_blade.custom_call";

class ConvertOperatorOp : public OpConversionPattern<OperatorOp> {
 public:
  using OpConversionPattern<OperatorOp>::OpConversionPattern;
  using OpAdaptor = typename OperatorOp::Adaptor;

  LogicalResult convertQuantizationRelatedOp(
      OperatorOp op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const {
    Location loc = op.getLoc();
    auto name = op.name();
    auto operands = adaptor.operands();
    auto input = operands[0];
    auto scale = operands[1];
    auto zeroPoint = operands[2];

#define I64_VAR_FROM_CONST_OPERAND(var, idx)                         \
  int64_t var;                                                       \
  if (!matchPattern(op.operands()[idx], m_TorchConstantInt(&var))) { \
    return op.emitError(#var " must be a scalar constant int");      \
  }
    I64_VAR_FROM_CONST_OPERAND(qmin, 3);
    I64_VAR_FROM_CONST_OPERAND(qmax, 4);
    I64_VAR_FROM_CONST_OPERAND(numBits, 5);
#undef I64_VAR_FROM_CONST_OPERAND

    SmallVector<int64_t, 4> axis;
    if (!matchPattern(op.operands()[6], m_TorchConstantIntList(axis))) {
      return op.emitError("only constant dims are supported a.t.m");
    }

#define BOOL_VAR_FROM_CONST_OPERAND(var, idx)                         \
  bool var;                                                           \
  if (!matchPattern(op.operands()[idx], m_TorchConstantBool(&var))) { \
    return op.emitError(#var " must be a scalar constant boolean");   \
  }

    BOOL_VAR_FROM_CONST_OPERAND(useSigned, 7);
    BOOL_VAR_FROM_CONST_OPERAND(useSymmetric, 8);
    BOOL_VAR_FROM_CONST_OPERAND(useDynamic, 9);
    BOOL_VAR_FROM_CONST_OPERAND(usePerChannel, 10);
#undef BOOL_VAR_FROM_CONST_OPERAND

    auto torchMlirResultTy =
        op.getResult(0).getType().dyn_cast<ValueTensorType>();
    auto resultTy = getTypeConverter()
                        ->convertType(torchMlirResultTy)
                        .dyn_cast<RankedTensorType>();
    if (!resultTy) {
      return op.emitError("failed to get type of output");
    }

    auto zeroPointTy = zeroPoint.getType().dyn_cast<RankedTensorType>();
    if (!zeroPointTy) {
      return op.emitError("zero point is not a RankedTensorType");
    }
    auto i32Ty = rewriter.getIntegerType(32);
    auto i64Ty = rewriter.getIntegerType(64);
    auto castedZeroPointTy =
        RankedTensorType::get(zeroPointTy.getShape(), i32Ty);
    Value castedZeroPoint =
        rewriter.create<mhlo::ConvertOp>(loc, castedZeroPointTy, zeroPoint);
    // Attributes ...
    auto axisAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({axis.size()}, rewriter.getIntegerType(64)),
        axis);
    auto numBitsAttr = rewriter.getIntegerAttr(i64Ty, numBits);
    auto qminAttr = rewriter.getIntegerAttr(i64Ty, qmin);
    auto qmaxAttr = rewriter.getIntegerAttr(i64Ty, qmax);
    auto useSignedAttr = rewriter.getBoolAttr(useSigned);
    auto useSymmetricAttr = rewriter.getBoolAttr(useSymmetric);
    auto useDynamicAttr = rewriter.getBoolAttr(useDynamic);
    // default round mode in torch is round-to-even.
    // TODO: should read it from the custom fake quant op.
    auto roundModeAttr = mlir::mhlo_disc::RoundModeEnumAttr::get(
        rewriter.getContext(), mlir::mhlo_disc::RoundModeEnum::RoundHalfToEven);
    Operation* newOutput;
    if (name == "torch_blade.fake_quant") {
      newOutput = rewriter.create<mhlo_disc::FakeQuantOp>(
          loc,
          resultTy,
          input,
          scale,
          castedZeroPoint,
          useSignedAttr,
          useSymmetricAttr,
          axisAttr,
          numBitsAttr,
          qminAttr,
          qmaxAttr,
          useDynamicAttr,
          roundModeAttr);
    } else if (name == "torch_blade.quantize") {
      newOutput = rewriter.create<mhlo_disc::QuantizeOp>(
          loc,
          resultTy,
          input,
          scale,
          castedZeroPoint,
          useSymmetricAttr,
          axisAttr,
          qminAttr,
          qmaxAttr,
          useDynamicAttr,
          roundModeAttr);
    } else if (name == "torch_blade.dequantize") {
      newOutput = rewriter.create<mhlo_disc::DequantizeOp>(
          loc,
          resultTy,
          input,
          scale,
          castedZeroPoint,
          useSymmetricAttr,
          axisAttr,
          useDynamicAttr,
          roundModeAttr);
    } else {
      return op.emitError("Unsupported kind of torch.operator.");
    }
    rewriter.replaceOp(op, newOutput->getResult(0));
    return success();
  }

  LogicalResult convertCustomCallOp(
      OperatorOp op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const {
    auto ctx = rewriter.getContext();
    SmallVector<Value> newOperands;
    Location loc = op.getLoc();
    auto operands = adaptor.operands();
    Value input = operands[0];
    Value weight = operands[1];
    Value bias = operands[2];
    newOperands.push_back(input);
    newOperands.push_back(weight);
    newOperands.push_back(bias);
    auto torchMlirResultTy =
        op.getResult(0).getType().dyn_cast<ValueTensorType>();
    auto resultTy = getTypeConverter()
                        ->convertType(torchMlirResultTy)
                        .dyn_cast<RankedTensorType>();
    const std::vector<std::string> requiredAttrName{
        "call_target_name",
        "device",
        "input_placements",
        "output_placements",
        "input_layouts",
        "output_layouts",
        "expected_input_layouts",
        "expected_output_layouts"};

    for (const auto& n : requiredAttrName) {
      if (!op->hasAttr(n)) {
        return op.emitError()
            << n
            << " attribute should be provided for torch_blade.custom_call.";
      }
    }

    StringAttr callTarget =
        op->getAttr("call_target_name").dyn_cast_or_null<StringAttr>();
    StringAttr device = op->getAttr("device").dyn_cast_or_null<StringAttr>();
    StringAttr inputPlacements =
        op->getAttr("input_placements").dyn_cast_or_null<StringAttr>();
    StringAttr outputPlacements =
        op->getAttr("output_placements").dyn_cast_or_null<StringAttr>();
    StringAttr inputLayouts =
        op->getAttr("input_layouts").dyn_cast_or_null<StringAttr>();
    StringAttr outputLayouts =
        op->getAttr("output_layouts").dyn_cast_or_null<StringAttr>();
    StringAttr expectedInputLayouts =
        op->getAttr("expected_input_layouts").dyn_cast_or_null<StringAttr>();
    StringAttr expectedOutputLayouts =
        op->getAttr("expected_output_layouts").dyn_cast_or_null<StringAttr>();

    DictionaryAttr customAttrs = DictionaryAttr::get(ctx, {});
    if (op->hasAttr("custom_attrs")) {
      customAttrs =
          op->getAttr("custom_attrs").dyn_cast_or_null<DictionaryAttr>();
    }
    Operation* newOutput = rewriter.create<mhlo_disc::CustomCallV2Op>(
        loc,
        resultTy,
        newOperands,
        callTarget,
        customAttrs,
        false, // has_side_effect
        device,
        inputPlacements,
        outputPlacements,
        inputLayouts,
        outputLayouts,
        expectedInputLayouts,
        expectedOutputLayouts);
    rewriter.replaceOp(op, newOutput->getResult(0));
    return success();
  }

  LogicalResult matchAndRewrite(
      OperatorOp op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    auto name = op.name();
    if (std::find(
            std::begin(quantizationOpList),
            std::end(quantizationOpList),
            name) != std::end(quantizationOpList)) {
      return convertQuantizationRelatedOp(op, adaptor, rewriter);
    } else if (name == customCallName) {
      return convertCustomCallOp(op, adaptor, rewriter);
    } else {
      auto emitter = op.emitError();
      emitter.append(
          "Currently, only the following types of OperatorOp are supported: ");
      for (const auto& s : quantizationOpList) {
        emitter.append(s).append(" ");
      }
      emitter.append(customCallName).append("\n");
      emitter.append("However, got: ");
      emitter.append(name);
      return emitter;
    }
  }
};

class DiscConvertTorchToDiscMhlo
    : public DiscConvertTorchToDiscMhloBase<DiscConvertTorchToDiscMhlo> {
  void runOnOperation() override {
    MLIRContext* context = &getContext();

    ConversionTarget target(*context);
    target.addLegalDialect<
        arith::ArithDialect,
        chlo::ChloDialect,
        mhlo::MhloDialect,
        mhlo_disc::MhloDiscDialect,
        tensor::TensorDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);
    patterns.add<ConvertOperatorOp>(typeConverter, context);
    target.addIllegalOp<OperatorOp>();

    if (failed(applyPartialConversion(
            getOperation(), target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} //  namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::torch::TorchConversion::
    createDiscConvertTorchToDiscMhlo() {
  return std::make_unique<DiscConvertTorchToDiscMhlo>();
}

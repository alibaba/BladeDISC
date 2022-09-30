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

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
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

class ConvertOperatorOp : public OpConversionPattern<OperatorOp> {
 public:
  using OpConversionPattern<OperatorOp>::OpConversionPattern;
  using OpAdaptor = typename OperatorOp::Adaptor;
  LogicalResult matchAndRewrite(
      OperatorOp op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    auto name = op.name();
    if ("torch_blade.fake_quant" == name) {
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

      auto resultTy = input.getType().dyn_cast<RankedTensorType>();
      if (!resultTy) {
        return op.emitError("failed to get type of input");
      }
      if (!resultTy.getElementType().isF32()) {
        return op.emitError(
            "torch_blade.fake_quant should have float32 as input type.");
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
        rewriter.getContext(), mlir::mhlo_disc::RoundModeEnum::RoundHalfToEven
      );
      Value newOp = rewriter.create<mhlo_disc::FakeQuantOp>(
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
      rewriter.replaceOp(op, {newOp});

      return success();
    }
    return op.emitError(
        "operators other than fake_quant are not supported a.t.m");
  }
};

class DiscConvertTorchToDiscMhlo
    : public DiscConvertTorchToDiscMhloBase<DiscConvertTorchToDiscMhlo> {
  void runOnOperation() override {
    MLIRContext* context = &getContext();

    ConversionTarget target(*context);
    target.addLegalDialect<
        arith::ArithmeticDialect,
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

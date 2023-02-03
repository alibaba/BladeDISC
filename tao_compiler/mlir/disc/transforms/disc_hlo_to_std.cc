/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/rewriters.h"

namespace mlir {
namespace disc_ral {

using mhlo::ComputeReshapeShapeOp;

namespace {

class ComputeReshapeShapeOpConverter
    : public OpConversionPattern<ComputeReshapeShapeOp> {
 public:
  using OpConversionPattern<ComputeReshapeShapeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ComputeReshapeShapeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override;
};

LogicalResult ComputeReshapeShapeOpConverter::matchAndRewrite(
    ComputeReshapeShapeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  Location loc = op.getLoc();
  MLIRContext* ctx = op->getContext();
  auto operands = adaptor.getOperands();
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value negOne = rewriter.create<arith::ConstantIndexOp>(loc, -1);

  Value numElements = operands[0];
  Value newShape = operands[1];
  ShapedType targetShapeType = newShape.getType().cast<ShapedType>();
  Type targetElemType = targetShapeType.getElementType();
  Type indexType = rewriter.getIndexType();

  if (!targetShapeType.hasStaticShape()) {
    op.emitError("only static rank is supported");
    return failure();
  }

  int64_t rank = targetShapeType.getDimSize(0);

  // in case there is a negOne in the new target shape values.
  Value accumNumElems = negOne;
  for (int64_t i = 0; i < rank; ++i) {
    Value idx = rewriter.create<arith::ConstantIndexOp>(loc, i);
    Value newShapeDimValue =
        rewriter.create<tensor::ExtractOp>(loc, newShape, idx);
    if (!targetElemType.isIndex()) {
      newShapeDimValue =
          rewriter.create<arith::IndexCastOp>(loc, indexType, newShapeDimValue);
    }
    accumNumElems =
        rewriter.create<arith::MulIOp>(loc, newShapeDimValue, accumNumElems);
  }

  // Handle following two corner cases:
  // - negOne in new target shape values
  // - original input tensor is a empty tensor.
  // In the above cases, accumNumElems is never used to calculate the result,
  // thus we simply set it to a safe value for division.
  Value isZero = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                numElements, zero);
  Value isNeg = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               accumNumElems, zero);
  Value isNegOrZero = rewriter.create<arith::OrIOp>(loc, isZero, isNeg);
  accumNumElems = rewriter.create<mlir::arith::SelectOp>(loc, isNegOrZero, one,
                                                         accumNumElems);

  SmallVector<Value, 4> extentValues;
  for (int64_t i = 0; i < rank; ++i) {
    Value idx = rewriter.create<arith::ConstantIndexOp>(loc, i);
    Value newShapeDimValue =
        rewriter.create<tensor::ExtractOp>(loc, newShape, idx);
    if (!targetElemType.isIndex()) {
      newShapeDimValue =
          rewriter.create<arith::IndexCastOp>(loc, indexType, newShapeDimValue);
    }
    Value isNegOne = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, newShapeDimValue, negOne);
    Value inferredDimValue =
        rewriter.create<arith::DivUIOp>(loc, numElements, accumNumElems);
    newShapeDimValue = rewriter.create<mlir::arith::SelectOp>(
        loc, isNegOne, inferredDimValue, newShapeDimValue);
    if (!targetElemType.isIndex())
      newShapeDimValue = rewriter.create<arith::IndexCastOp>(
          loc, targetElemType, newShapeDimValue);
    extentValues.push_back(newShapeDimValue);
  }

  // Materialize extent tensor.
  Value staticExtentTensor =
      rewriter.create<tensor::FromElementsOp>(loc, extentValues);
  rewriter.replaceOp(op, ValueRange{staticExtentTensor});

  return success();
}

class ConvertHloToStandardPass
    : public ConvertHloToStandardPassBase<ConvertHloToStandardPass> {
  void runOnOperation() override;
};

void ConvertHloToStandardPass::runOnOperation() {
  // Setup target legality.
  MLIRContext& ctx = getContext();
  ConversionTarget target(ctx);
  target.addLegalDialect<tensor::TensorDialect, arith::ArithDialect>();
  target.addIllegalOp<ComputeReshapeShapeOp>();

  // Setup conversion patterns.
  RewritePatternSet patterns(&ctx);
  // clang-format: off
  patterns.insert<ComputeReshapeShapeOpConverter>(&ctx);
  // clang-format: on

  // Apply conversion.
  func::FuncOp func = getOperation();
  if (failed(applyPartialConversion(func, target, std::move(patterns))))
    signalPassFailure();
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createDiscConvertHloToStandardPass() {
  return std::make_unique<ConvertHloToStandardPass>();
}

}  // namespace disc_ral
}  // namespace mlir

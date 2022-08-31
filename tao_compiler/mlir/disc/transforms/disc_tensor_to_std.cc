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

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"
#include "tensorflow/compiler/mlir/disc/transforms/rewriters.h"

namespace mlir {
namespace disc_ral {

using tensor::GenerateOp;

namespace {

class GenerateOpConverter : public OpConversionPattern<GenerateOp> {
 public:
  using OpConversionPattern<GenerateOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      GenerateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override;
};

LogicalResult GenerateOpConverter::matchAndRewrite(
    GenerateOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto resultTy = op.getType().dyn_cast<RankedTensorType>();
  if (!resultTy || !resultTy.hasStaticShape()) {
    op.emitError("only static shape ranked tensor result type is supported");
    return failure();
  }

  // for now, only rank-1 tensor is supported.
  if (resultTy.getRank() != 1) {
    op.emitError("only rank-1 result tensor is supported");
    return failure();
  }

  // for now, only single block region is supported.
  if (op.getBody().getBlocks().size() != 1) {
    op.emitError("only single block region inside generate op is supported");
    return failure();
  }

  Location loc = op.getLoc();
  ImplicitLocOpBuilder lb(loc, rewriter);

  Block& block = op.getBody().front();
  int64_t numElems = resultTy.getDimSize(0);
  SmallVector<Value, 4> extentValues;
  for (int64_t i = 0; i < numElems; ++i) {
    Value idx = lb.create<arith::ConstantIndexOp>(i);
    BlockAndValueMapping mapping;
    mapping.map(block.getArgument(0), idx);
    for (Operation& op : block.without_terminator()) {
      lb.clone(op, mapping);
    }
    extentValues.push_back(
        mapping.lookup(block.getTerminator()->getOperand(0)));
  }

  // Materialize extent tensor.
  Value staticExtentTensor =
      rewriter.create<tensor::FromElementsOp>(loc, extentValues);
  rewriter.replaceOp(op, ValueRange{staticExtentTensor});

  return success();
}

class ConvertTensorToStandardPass
    : public ConvertTensorToStandardPassBase<ConvertTensorToStandardPass> {
  void runOnOperation() override;
};

void ConvertTensorToStandardPass::runOnOperation() {
  // Setup target legality.
  MLIRContext& ctx = getContext();
  ConversionTarget target(ctx);
  target.addLegalDialect<arith::ArithmeticDialect>();
  target.addLegalOp<func::FuncOp, ModuleOp>();
  target.addLegalOp<tensor::ExtractOp, tensor::DimOp, tensor::FromElementsOp>();
  target.addIllegalOp<GenerateOp>();

  // Setup conversion patterns.
  RewritePatternSet patterns(&ctx);
  // clang-format: off
  patterns.insert<GenerateOpConverter>(&ctx);
  // clang-format: on

  // Apply conversion.
  func::FuncOp func = getOperation();
  if (failed(applyPartialConversion(func, target, std::move(patterns))))
    signalPassFailure();
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createDiscConvertTensorToStandardPass() {
  return std::make_unique<ConvertTensorToStandardPass>();
}

}  // namespace disc_ral
}  // namespace mlir

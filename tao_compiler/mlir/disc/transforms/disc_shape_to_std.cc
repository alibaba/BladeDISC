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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
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
#include "mlir/disc/IR/disc_shape_ops.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/rewriters.h"

namespace mlir {
namespace disc_ral {

namespace {

using disc_shape::DelinearizeOp;
using disc_shape::LinearizeOp;
using shape::BroadcastOp;
using shape::ConcatOp;
using shape::ConstShapeOp;
using shape::NumElementsOp;
using shape::ShapeOfOp;
using shape::ShapeType;
using shape::SplitAtOp;
using shape::ToExtentTensorOp;

Value getDimSize(Value v, int dim, ImplicitLocOpBuilder lb) {
  if (v.getType().isa<RankedTensorType>())
    return lb.create<tensor::DimOp>(v, dim);
  assert(v.getType().isa<MemRefType>());
  return lb.create<memref::DimOp>(v, dim);
}

class ShapeOfOpConversion : public OpConversionPattern<ShapeOfOp> {
 public:
  using OpConversionPattern<ShapeOfOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ShapeOfOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override;
};

LogicalResult ShapeOfOpConversion::matchAndRewrite(
    ShapeOfOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  // For now, only error-free types are supported by this lowering.
  if (op.getType().isa<ShapeType>()) return failure();

  Location loc = op.getLoc();
  ImplicitLocOpBuilder lb(loc, rewriter);
  Value arg = adaptor.getArg();
  auto argTy = arg.getType().dyn_cast<ShapedType>();

  // Only ranked operand is supported.
  if (!argTy || !argTy.hasRank()) return failure();

  // Build values for individual extents.
  SmallVector<Value, 8> extentValues;
  int64_t rank = argTy.getRank();
  for (int64_t i = 0; i < rank; i++) {
    if (argTy.isDynamicDim(i)) {
      extentValues.push_back(getDimSize(arg, i, lb));
    } else {
      Value extent =
          rewriter.create<arith::ConstantIndexOp>(loc, argTy.getDimSize(i));
      extentValues.push_back(extent);
    }
  }

  // Materialize extent tensor.
  Value staticExtentTensor =
      rewriter.create<tensor::FromElementsOp>(loc, extentValues);
  if (staticExtentTensor.getType() != op.getType() &&
      !op.getType().isa<ShapeType>()) {
    staticExtentTensor =
        rewriter.create<tensor::CastOp>(loc, op.getType(), staticExtentTensor);
  }
  rewriter.replaceOp(op, ValueRange{staticExtentTensor});

  return success();
}

class ConstShapeOpConverter : public OpConversionPattern<ConstShapeOp> {
 public:
  using OpConversionPattern<ConstShapeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ConstShapeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override;
};

LogicalResult ConstShapeOpConverter::matchAndRewrite(
    ConstShapeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto loc = op.getLoc();
  SmallVector<Value, 4> extentOperands;
  for (auto extent : op.getShape()) {
    extentOperands.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, extent.getLimitedValue()));
  }
  Type resultTy =
      RankedTensorType::get({op.getShape().size()}, rewriter.getIndexType());
  Value tensor =
      rewriter.create<tensor::FromElementsOp>(loc, resultTy, extentOperands);
  rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultTy, tensor);
  return success();
}

struct BroadcastOpConverter : public OpConversionPattern<BroadcastOp> {
  using OpConversionPattern<BroadcastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      BroadcastOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override;
};

// Get the resulting extent in a given dimension. This is computed with any
// number of extent tensors and shifted offsets into them.
Value getBroadcastedDim(ImplicitLocOpBuilder lb, ValueRange extentTensors,
                        SmallVectorImpl<int64_t>& rankDiffs,
                        int64_t outputDimension) {
  Value one = lb.create<arith::ConstantIndexOp>(1);
  Value broadcastedDim = one;
  for (auto tup : llvm::zip(extentTensors, rankDiffs)) {
    Value shape = std::get<0>(tup);
    int64_t rankDiff = std::get<1>(tup);
    if (outputDimension < rankDiff) {
      continue;
    }

    Value lesserRankOperandDimension =
        lb.create<arith::ConstantIndexOp>(outputDimension - rankDiff);
    Value lesserRankOperandExtent = lb.create<tensor::ExtractOp>(
        shape, ValueRange{lesserRankOperandDimension});
    Value dimIsOne = lb.create<arith::CmpIOp>(arith::CmpIPredicate::eq,
                                              lesserRankOperandExtent, one);
    broadcastedDim = lb.create<mlir::arith::SelectOp>(dimIsOne, broadcastedDim,
                                                      lesserRankOperandExtent);
  }
  return broadcastedDim;
}

LogicalResult BroadcastOpConverter::matchAndRewrite(
    BroadcastOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  // For now, this lowering is only defined on `tensor<?xindex>` operands, not
  // on shapes.
  if (llvm::any_of(adaptor.getOperands(),
                   [](Value v) { return v.getType().isa<ShapeType>(); }))
    return failure();

  Location loc = op.getLoc();
  ImplicitLocOpBuilder lb(loc, rewriter);

  Value zero = lb.create<arith::ConstantIndexOp>(0);
  Type indexTy = lb.getIndexType();

  // Save all the ranks for bounds checking. Because this is a tensor
  // representing the shape extents, the rank is the extent of the only
  // dimension in the tensor.
  SmallVector<int64_t> ranks, rankDiffs;
  for (Value shape : adaptor.getShapes()) {
    auto shapeTy = shape.getType().dyn_cast<RankedTensorType>();
    if (!shapeTy || !shapeTy.hasStaticShape()) return failure();
    ranks.push_back(shapeTy.getDimSize(0));
  }

  // Find the maximum rank
  int64_t maxRank = ranks.front();
  for (int64_t v : llvm::drop_begin(ranks, 1)) {
    if (v > maxRank) maxRank = v;
  }

  // Calculate the difference of ranks and the maximum rank for later offsets.
  for (int64_t v : ranks) rankDiffs.push_back(maxRank - v);

  SmallVector<Value, 4> extentValues;
  for (int64_t i = 0; i < maxRank; ++i) {
    extentValues.push_back(
        getBroadcastedDim(lb, adaptor.getShapes(), rankDiffs, i));
  }

  // Materialize extent tensor.
  Value staticExtentTensor =
      rewriter.create<tensor::FromElementsOp>(loc, extentValues);

  if (staticExtentTensor.getType() != op.getType() &&
      !op.getType().isa<ShapeType>()) {
    staticExtentTensor =
        rewriter.create<tensor::CastOp>(loc, op.getType(), staticExtentTensor);
  }
  rewriter.replaceOp(op, ValueRange{staticExtentTensor});

  return success();
}

class SplitAtOpConversion : public OpConversionPattern<SplitAtOp> {
 public:
  using OpConversionPattern<SplitAtOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SplitAtOp op, OpAdaptor adapter,
      ConversionPatternRewriter& rewriter) const override;
};

LogicalResult SplitAtOpConversion::matchAndRewrite(
    SplitAtOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  Value operand = adaptor.getOperand();
  auto operandTy = operand.getType().dyn_cast<RankedTensorType>();
  if (!operandTy || !operandTy.hasStaticShape()) return failure();

  Value index = adaptor.getIndex();
  auto indexOp =
      dyn_cast_or_null<arith::ConstantIndexOp>(index.getDefiningOp());
  if (!indexOp) return failure();

  Location loc = op.getLoc();
  ImplicitLocOpBuilder lb(loc, rewriter);

  int64_t rank = operandTy.getDimSize(0);
  int64_t indexVal = indexOp.getValue().cast<IntegerAttr>().getInt();
  if (indexVal < 0) indexVal += rank;

  SmallVector<Value, 4> headExtentValues;
  SmallVector<Value, 4> tailExtentValues;
  for (int64_t i = 0; i < rank; ++i) {
    Value idx = lb.create<arith::ConstantIndexOp>(i);
    Value dimSize = lb.create<tensor::ExtractOp>(operand, idx);
    if (i < indexVal)
      headExtentValues.push_back(dimSize);
    else
      tailExtentValues.push_back(dimSize);
  }

  // Materialize extent tensor.
  Value headExtentTensor =
      rewriter.create<tensor::FromElementsOp>(loc, headExtentValues);
  Value tailExtentTensor =
      rewriter.create<tensor::FromElementsOp>(loc, tailExtentValues);

  rewriter.replaceOp(op, {headExtentTensor, tailExtentTensor});
  return success();
}

class ToExtentTensorOpConversion
    : public OpConversionPattern<ToExtentTensorOp> {
 public:
  using OpConversionPattern<ToExtentTensorOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ToExtentTensorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    if (!adaptor.getInput().getType().isa<RankedTensorType>())
      return rewriter.notifyMatchFailure(op, "input needs to be a tensor");

    Value replaceValue = adaptor.getInput();
    rewriter.replaceOp(op, ValueRange{replaceValue});
    return success();
  }
};

class ConcatOpConversion : public OpConversionPattern<ConcatOp> {
 public:
  using OpConversionPattern<ConcatOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ConcatOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override;
};

LogicalResult ConcatOpConversion::matchAndRewrite(
    ConcatOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const {
  Value lhs = adaptor.getLhs();
  Value rhs = adaptor.getRhs();

  auto lhsTy = lhs.getType().dyn_cast<RankedTensorType>();
  auto rhsTy = rhs.getType().dyn_cast<RankedTensorType>();
  if (!lhsTy || !lhsTy.hasStaticShape() || !rhsTy || !rhsTy.hasStaticShape())
    return failure();

  int64_t lhsRank = lhsTy.getDimSize(0);
  int64_t rhsRank = rhsTy.getDimSize(0);

  SmallVector<Value, 4> extentValues;
  for (int64_t i = 0; i < lhsRank; ++i) {
    Value idx = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), i);
    extentValues.push_back(
        rewriter.create<tensor::ExtractOp>(op.getLoc(), lhs, idx));
  }

  for (int64_t i = 0; i < rhsRank; ++i) {
    Value idx = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), i);
    extentValues.push_back(
        rewriter.create<tensor::ExtractOp>(op.getLoc(), rhs, idx));
  }

  // Materialize extent tensor.
  Value extentTensor =
      rewriter.create<tensor::FromElementsOp>(op.getLoc(), extentValues);

  rewriter.replaceOp(op, {extentTensor});
  return success();
}

class NumElementsOpConversion : public OpConversionPattern<NumElementsOp> {
 public:
  using OpConversionPattern<NumElementsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      NumElementsOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override;
};

LogicalResult NumElementsOpConversion::matchAndRewrite(
    NumElementsOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  Value shapeTensor = adaptor.getOperands()[0];
  auto shapeType = shapeTensor.getType().dyn_cast<RankedTensorType>();
  if (!shapeType || !shapeType.hasStaticShape()) {
    op.emitError("only static shape ranked tensor result type is supported");
    return failure();
  }

  Location loc = op.getLoc();
  int64_t rank = shapeType.getDimSize(0);
  Value numElems = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  for (int64_t i = 0; i < rank; ++i) {
    Value idx = rewriter.create<arith::ConstantIndexOp>(loc, i);
    Value dimValue = rewriter.create<tensor::ExtractOp>(loc, shapeTensor, idx);
    numElems = rewriter.create<arith::MulIOp>(loc, dimValue, numElems);
  }

  rewriter.replaceOp(op, ValueRange{numElems});

  return success();
}

class LinearizeOpConversion : public OpConversionPattern<LinearizeOp> {
 public:
  using OpConversionPattern<LinearizeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      LinearizeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override;
};

LogicalResult LinearizeOpConversion::matchAndRewrite(
    LinearizeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  Location loc = op.getLoc();
  int rank = adaptor.getMultiDimIndexes().size();
  Value linear = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  ;
  if (rank > 0) {
    linear = adaptor.getMultiDimIndexes().front();
    for (auto&& z : llvm::zip(adaptor.getMultiDimIndexes().drop_front(),
                              adaptor.getShapeDimIndexes().drop_front())) {
      linear = rewriter.create<arith::AddIOp>(
          loc, rewriter.create<arith::MulIOp>(loc, linear, std::get<1>(z)),
          std::get<0>(z));
    }
  }

  rewriter.replaceOp(op, ValueRange{linear});
  return success();
}

class DelinearizeOpConversion : public OpConversionPattern<DelinearizeOp> {
 public:
  using OpConversionPattern<DelinearizeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      DelinearizeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override;
};

LogicalResult DelinearizeOpConversion::matchAndRewrite(
    DelinearizeOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  Location loc = op.getLoc();
  int rank = adaptor.getShapeDimIndexes().size();
  if (rank == 0) {
    rewriter.eraseOp(op);
  } else if (rank == 1) {
    rewriter.replaceOp(op, ValueRange{adaptor.getLinearIndex()});
  } else {
    Value linear = adaptor.getLinearIndex();
    SmallVector<Value> multiDims(rank);
    for (auto&& en : llvm::enumerate(
             llvm::reverse(adaptor.getShapeDimIndexes().drop_front()))) {
      multiDims[rank - 1 - en.index()] =
          rewriter.create<arith::RemUIOp>(loc, linear, en.value());
      linear = rewriter.create<arith::DivUIOp>(loc, linear, en.value());
    }
    multiDims[0] = linear;
    rewriter.replaceOp(op, multiDims);
  }
  return success();
}

class ConvertShapeToStandardPass
    : public ConvertShapeToStandardPassBase<ConvertShapeToStandardPass> {
  void runOnOperation() override;
};

void ConvertShapeToStandardPass::runOnOperation() {
  // Setup target legality.
  MLIRContext& ctx = getContext();
  ConversionTarget target(ctx);
  target.addLegalDialect<arith::ArithDialect, tensor::TensorDialect>();
  target.addLegalOp<func::FuncOp, ModuleOp>();

  // Setup conversion patterns.
  RewritePatternSet patterns(&ctx);
  // clang-format: off
  patterns
      .insert<BroadcastOpConverter, ConcatOpConversion, NumElementsOpConversion,
              ShapeOfOpConversion, SplitAtOpConversion,
              ToExtentTensorOpConversion, LinearizeOpConversion,
              DelinearizeOpConversion, ConstShapeOpConverter>(&ctx);
  // clang-format: on

  // Apply conversion.
  func::FuncOp func = getOperation();
  if (failed(applyPartialConversion(func, target, std::move(patterns))))
    signalPassFailure();
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createDiscConvertShapeToStandardPass() {
  return std::make_unique<ConvertShapeToStandardPass>();
}

}  // namespace disc_ral
}  // namespace mlir

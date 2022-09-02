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

// This file implements logic for lowering HLO DISC dialect to LHLO DISC
// dialect.

#include <algorithm>
#include <utility>

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tensorflow/compiler/mlir/disc/IR/disc_shape_ops.h"
#include "tensorflow/compiler/mlir/disc/IR/hlo_disc_ops.h"
#include "tensorflow/compiler/mlir/disc/IR/lhlo_disc_ops.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"
#include "tensorflow/compiler/mlir/disc/transforms/codegen_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/disc_map_hlo_to_lhlo_op.h"
#include "tensorflow/compiler/mlir/disc/transforms/rewriters.h"

namespace mlir {
namespace mhlo_disc {
namespace {

using disc_shape::TieShapeOp;

template <typename T>
using BaseOpConversion = OpConversionPattern<T>;

Value InsertDynamicAlloc(Location loc, Value result, Value shape_operand,
                         ConversionPatternRewriter* rewriter) {
  auto result_type = result.getType().dyn_cast<RankedTensorType>();
  if (!result_type) {
    result.getDefiningOp()->emitOpError()
        << "tensor to buffer conversion expects ranked results";
  }
  auto memref_type =
      MemRefType::get(result_type.getShape(), result_type.getElementType());

  // Extract the required element out of the vector.
  SmallVector<Value, 4> dynamic_operands;
  for (auto shape_element : llvm::enumerate(result_type.getShape())) {
    if (shape_element.value() != ShapedType::kDynamicSize) continue;
    Value index =
        rewriter->create<arith::ConstantIndexOp>(loc, shape_element.index());
    Value alloc_operand =
        rewriter->create<tensor::ExtractOp>(loc, shape_operand, index);
    if (!alloc_operand.getType().isIndex()) {
      alloc_operand = rewriter->create<arith::IndexCastOp>(
          loc, rewriter->getIndexType(), alloc_operand);
    }
    dynamic_operands.push_back(alloc_operand);
  }

  return rewriter->create<memref::AllocOp>(loc, memref_type, dynamic_operands);
}

Value InsertAlloc(Location loc, OpResult result,
                  ConversionPatternRewriter* rewriter) {
  auto result_type = result.getType().dyn_cast<RankedTensorType>();
  if (!result_type || !result_type.hasStaticShape()) {
    result.getDefiningOp()->emitOpError()
        << "tensor to buffer conversion expects statically shaped results";
  }
  auto memref_type =
      MemRefType::get(result_type.getShape(), result_type.getElementType());
  OpBuilder::InsertionGuard guard(*rewriter);
  rewriter->setInsertionPoint(result.getDefiningOp());
  return rewriter->create<memref::AllocOp>(loc, memref_type);
}

/// Converts the results of the operation `op` to memref types and append them
/// to the `results` vector.
LogicalResult ConvertResults(Operation* op, SmallVectorImpl<Value>& results,
                             ConversionPatternRewriter& rewriter) {
  size_t num_operands = results.size();
  SmallVector<Value, 2> tensor_operands;
  for (auto result : llvm::enumerate(op->getResults())) {
    RankedTensorType resultType =
        result.value().getType().dyn_cast<RankedTensorType>();
    if (!resultType) return failure();

    if (resultType.hasStaticShape()) {
      results.push_back(InsertAlloc(op->getLoc(), result.value(), &rewriter));
      continue;
    }
    auto shape_type_op = dyn_cast<InferShapedTypeOpInterface>(op);
    if (!shape_type_op) return failure();

    if (tensor_operands.empty()) {
      for (auto operand : ArrayRef<Value>(results).take_front(num_operands)) {
        auto operand_type = operand.getType().dyn_cast<MemRefType>();
        if (!operand_type) return failure();
        tensor_operands.push_back(rewriter.create<bufferization::ToTensorOp>(
            op->getLoc(),
            RankedTensorType::get(operand_type.getShape(),
                                  operand_type.getElementType()),
            operand));
      }
    }

    SmallVector<Value, 1> results_shape;
    auto status = shape_type_op.reifyReturnTypeShapes(rewriter, tensor_operands,
                                                      results_shape);
    if (failed(status)) return failure();
    results.push_back(InsertDynamicAlloc(op->getLoc(), result.value(),
                                         results_shape[result.index()],
                                         &rewriter));
  }
  return success();
}

template <typename HloOpTy>
class HloToLhloOpConverter : public BaseOpConversion<HloOpTy> {
 public:
  using BaseOpConversion<HloOpTy>::BaseOpConversion;
  LogicalResult matchAndRewrite(
      HloOpTy hloOp, typename HloOpTy::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Operation* op = hloOp.getOperation();
    auto operands = adaptor.getOperands();
    SmallVector<Value, 4> buffer_args(operands.begin(), operands.end());
    if (failed(ConvertResults(op, buffer_args, rewriter))) return failure();
    rewriter.create<mhlo_disc::HloToLhloOp<HloOpTy>>(
        op->getLoc(), llvm::None, buffer_args, op->getAttrs());
    rewriter.replaceOp(
        op, llvm::makeArrayRef(buffer_args).drop_front(operands.size()));
    return success();
  }
};

struct HloToLhloCustomCallOpConverter : public BaseOpConversion<CustomCallOp> {
 public:
  using BaseOpConversion<CustomCallOp>::BaseOpConversion;

  LogicalResult matchAndRewrite(
      CustomCallOp hloOp, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Operation* op = hloOp.getOperation();
    auto operands = adaptor.getOperands();
    SmallVector<Value, 2> buffer_args(operands.begin(), operands.end());
    if (failed(ConvertResults(op, buffer_args, rewriter))) return failure();

    auto lhloOp = rewriter.create<lmhlo_disc::CustomCallOp>(
        op->getLoc(), llvm::None, buffer_args, op->getAttrs());
    // Setup AttrSizedOperandSegments attribute to indicate number of operands
    // for args and outputs.
    const int32_t segments[2] = {static_cast<int32_t>(operands.size()),
                                 static_cast<int32_t>(op->getNumResults())};
    auto attrValue = mlir::DenseI32ArrayAttr::get(op->getContext(), segments);
    lhloOp->setAttr(lhloOp.getOperandSegmentSizeAttr(), attrValue);

    rewriter.replaceOp(op, ArrayRef<Value>(buffer_args).slice(operands.size()));
    return success();
  }
};

struct TieShapeOpConverter : public BaseOpConversion<TieShapeOp> {
 public:
  using BaseOpConversion<TieShapeOp>::BaseOpConversion;

  LogicalResult matchAndRewrite(
      TieShapeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    auto operands = adaptor.getOperands();
    Value memref = operands[0];
    int64_t rank = operands.size() - 1;
    auto memrefTy = memref.getType().cast<MemRefType>();
    assert(memrefTy.getRank() == rank);

    Value castedValue = disc_ral::CastMemRefTo(rewriter, loc, memref, memrefTy,
                                               operands.drop_front());
    StringRef attrName = disc_shape::SymbolicDimOp::getSymbolicDimAttrName();
    if (op->hasAttr(attrName)) {
      castedValue.getDefiningOp()->setAttr(attrName, op->getAttr(attrName));
    }
    rewriter.replaceOp(op, {castedValue});
    return success();
  }
};

// TODO(lanbo.llb): Support there is one -1 in new_shape
struct SparseReshapeOpConverter
    : public BaseOpConversion<mhlo_disc::SparseReshapeOp> {
 public:
  using BaseOpConversion<mhlo_disc::SparseReshapeOp>::BaseOpConversion;

  LogicalResult matchAndRewrite(
      mhlo_disc::SparseReshapeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto loc = op.getLoc();
    auto operands = adaptor.getOperands();

    SmallVector<Value, 2> buffer_args(operands.begin(), operands.end());
    if (failed(ConvertResults(op, buffer_args, rewriter))) return failure();
    auto sparse_reshape_op = rewriter.create<lmhlo_disc::SparseReshapeOp>(
        loc, llvm::None, buffer_args);
    rewriter.replaceOp(op, ArrayRef<Value>(buffer_args).slice(operands.size()));
    return success();
  }
};

struct DiscHloLegalizeToLhlo
    : public DiscHloLegalizeToLhloPassBase<DiscHloLegalizeToLhlo> {
  using DiscHloLegalizeToLhloPassBase<
      DiscHloLegalizeToLhlo>::DiscHloLegalizeToLhloPassBase;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<lmhlo_disc::LmhloDiscDialect, memref::MemRefDialect,
                    shape::ShapeDialect>();
  }

 public:
  DiscHloLegalizeToLhlo() = default;

  void runOnOperation() override {
    auto& context = getContext();
    RewritePatternSet patterns(&context);
    ConversionTarget target(context);
    target.addLegalDialect<
        arith::ArithmeticDialect, lmhlo_disc::LmhloDiscDialect,
        bufferization::BufferizationDialect, memref::MemRefDialect,
        shape::ShapeDialect, tensor::TensorDialect>();
    target.addIllegalDialect<mhlo_disc::MhloDiscDialect>();
    target.addIllegalOp<disc_shape::TieShapeOp>();

    bufferization::BufferizeTypeConverter converter;
    populateDiscHLOToLHLOConversionPattern(&context, &converter, &patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
}  // namespace

void populateDiscHLOToLHLOConversionPattern(
    MLIRContext* context, bufferization::BufferizeTypeConverter* converter,
    RewritePatternSet* patterns) {
  // clang-format off
  patterns->insert<
      HloToLhloOpConverter<mhlo_disc::H2DOp>,
      HloToLhloOpConverter<mhlo_disc::D2HOp>,
      HloToLhloCustomCallOpConverter,
      TieShapeOpConverter,
      SparseReshapeOpConverter
  >(*converter, context);
  // clang-format on
}

std::unique_ptr<OperationPass<ModuleOp>> createDiscLegalizeToLhloPass() {
  return std::make_unique<DiscHloLegalizeToLhlo>();
}

}  // namespace mhlo_disc
}  // namespace mlir

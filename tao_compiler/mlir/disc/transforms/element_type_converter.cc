/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// A pass that eliminates certain element types as the input or output of ops by
// inserting Convert ops. This allows a backend to support an element type while
// only actually implementing the Convert op for that element type. This is
// generally not the fastest approach, but it works.

#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"                // from @llvm-project
#include "mlir/IR/Attributes.h"                          // from @llvm-project
#include "mlir/IR/Location.h"                            // from @llvm-project
#include "mlir/IR/MLIRContext.h"                         // from @llvm-project
#include "mlir/IR/Operation.h"                           // from @llvm-project
#include "mlir/IR/PatternMatch.h"                        // from @llvm-project
#include "mlir/Pass/Pass.h"                              // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"                      // from @llvm-project
#include "mlir/disc/transforms/PassDetail.h"

namespace mlir {
namespace disc_ral {

namespace {

template <typename Op>
static void BuildReduceBody(Type element_type, Region* body,
                            OpBuilder* builder) {
  OpBuilder::InsertionGuard guard(*builder);
  Block* block = builder->createBlock(body);

  // Block arguments are scalars of the given element type.
  RankedTensorType type = RankedTensorType::get(/*shape=*/{}, element_type);
  Location loc = body->getLoc();
  block->addArgument(type, loc);
  block->addArgument(type, loc);

  Op reducer =
      builder->create<Op>(loc, block->getArgument(0), block->getArgument(1));
  builder->create<mhlo::ReturnOp>(loc, reducer.getResult());
}

mhlo::ReduceOp CloneAndReplaceElementType(mhlo::ReduceOp op,
                                          PatternRewriter& rewriter,
                                          Type elem_type,
                                          ArrayRef<Value> operands) {
  int num_non_return_hlo_ops = 0;
  Operation* map_op = nullptr;
  op.getBody().walk([&](Operation* hlo_op) {
    if (isa<mhlo::ReturnOp>(hlo_op)) return;
    ++num_non_return_hlo_ops;
    map_op = hlo_op;
  });

  assert(num_non_return_hlo_ops == 1 && map_op &&
         "Not support reduction region");

  Location loc = op.getLoc();
  mhlo::ReduceOp new_op = rewriter.create<mhlo::ReduceOp>(
      loc, operands[0], operands[1], op.getDimensions());

  if (isa<mhlo::AddOp>(map_op)) {
    BuildReduceBody<mhlo::AddOp>(elem_type, &new_op.getBody(), &rewriter);
  } else if (isa<mhlo::MulOp>(map_op)) {
    BuildReduceBody<mhlo::MulOp>(elem_type, &new_op.getBody(), &rewriter);
  } else if (isa<mhlo::MaxOp>(map_op)) {
    BuildReduceBody<mhlo::MaxOp>(elem_type, &new_op.getBody(), &rewriter);
  } else if (isa<mhlo::MinOp>(map_op)) {
    BuildReduceBody<mhlo::MinOp>(elem_type, &new_op.getBody(), &rewriter);
  } else if (isa<mhlo::AndOp>(map_op)) {
    // We use MinOp to replace AndOp since std AndOp requires operands have i1
    // type.
    BuildReduceBody<mhlo::MinOp>(elem_type, &new_op.getBody(), &rewriter);
  } else if (isa<mhlo::OrOp>(map_op)) {
    // We use MaxOp to replace OrOp since std AndOp requires operands have i1
    // type.
    BuildReduceBody<mhlo::MaxOp>(elem_type, &new_op.getBody(), &rewriter);
  } else {
    assert(false && "not supported reduce type");
  }

  new_op->setAttrs(op->getAttrs());

  return new_op;
}

struct ConvertReduceOpWithSmallWidthIntType
    : public OpRewritePattern<mhlo::ReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::ReduceOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getNumOperands() > 2) {
      // Currently do not support Variadic operand number.
      return failure();
    }

    SmallVector<int64_t, 4> dims_to_reduce;
    for (auto v : op.getDimensions().getValues<APInt>()) {
      dims_to_reduce.push_back(v.getSExtValue());
    }
    RankedTensorType ty =
        op.getOperand(0).getType().dyn_cast<RankedTensorType>();
    assert(ty && "suppose reduce input is a ranked tensor");
    int rank = ty.getRank();
    int ndims_to_reduce = static_cast<int>(dims_to_reduce.size());

    if (rank != 2 || ndims_to_reduce != 1) {
      // Suppose that there are only rank-2 row/colunm reduction after
      // `canonicalize-reduction` pass.
      return failure();
    }

    // Currently we convert column reduce ops having type i1, i8 or i16 to i32
    // since our codegen engine does not support column reduction ATM.
    //
    // To be concrete, we currently rely on atomic ops to implement column
    // reduction. On Nvidia GPUs, atomic op can only operate on 32 bit and 64
    // bit integers. In XLA, if the element type of the binary operation is
    // smaller than 32 bits, int32 is used to wrapper original element type.
    // The implementation rely on `atomicCAS` with is not accessible in the
    // current MLIR version.
    auto int_tp = ty.getElementType().dyn_cast<IntegerType>();
    if (!int_tp || int_tp.getWidth() >= 32) {
      return failure();
    }

    Location loc = op.getLoc();
    IntegerType elem_type = rewriter.getIntegerType(32);
    SmallVector<Value, 4> converted_operands;
    for (auto value : op.getOperands()) {
      converted_operands.push_back(
          rewriter.create<mhlo::ConvertOp>(loc, value, elem_type));
    }

    mhlo::ReduceOp new_op =
        CloneAndReplaceElementType(op, rewriter, elem_type, converted_operands);
    assert(new_op && "convert element type of reduce op failed");

    SmallVector<Value, 4> converted_results;
    for (const auto& z : llvm::zip(new_op.getResults(), op.getResults())) {
      converted_results.push_back(rewriter.create<mhlo::ConvertOp>(
          loc, std::get<0>(z), ty.getElementType()));
      converted_results.back().setType(std::get<1>(z).getType());
    }

    rewriter.replaceOp(op, converted_results);
    return success();
  }
};

template <typename OpTy>
struct ConvertConvOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    FloatType f16_ty = rewriter.getF16Type();
    FloatType f32_ty = rewriter.getF32Type();
    RankedTensorType lhs_ty = lhs.getType().dyn_cast<RankedTensorType>();
    RankedTensorType rhs_ty = rhs.getType().dyn_cast<RankedTensorType>();
    RankedTensorType result_ty =
        op.getType().template dyn_cast<RankedTensorType>();
    if (!lhs_ty || !rhs_ty || lhs_ty.getElementType() != f32_ty ||
        rhs_ty.getElementType() != f32_ty) {
      return failure();
    }

    Value lhs_f16 = rewriter.create<mhlo::ConvertOp>(loc, lhs, f16_ty);
    Value rhs_f16 = rewriter.create<mhlo::ConvertOp>(loc, rhs, f16_ty);
    RankedTensorType f16_tensor_ty =
        RankedTensorType::getChecked(loc, result_ty.getShape(), f16_ty);
    SmallVector<Value, 4> newOperands = {lhs_f16, rhs_f16};
    for (int i = 2; i < op->getOperands().size(); ++i) {
      newOperands.push_back(op->getOperand(i));
    }
    Value conv =
        rewriter.create<OpTy>(loc, f16_tensor_ty, newOperands, op->getAttrs());
    Value fp32_conv = rewriter.create<mhlo::ConvertOp>(loc, conv, f32_ty);
    fp32_conv.setType(op.getResult().getType());
    rewriter.replaceOp(op, fp32_conv);
    return success();
  }
};

struct ConvertDotGeneralOp : public OpRewritePattern<mhlo::DotGeneralOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::DotGeneralOp op,
                                PatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    FloatType f16_ty = rewriter.getF16Type();
    FloatType f32_ty = rewriter.getF32Type();
    RankedTensorType lhs_ty = lhs.getType().dyn_cast<RankedTensorType>();
    RankedTensorType rhs_ty = rhs.getType().dyn_cast<RankedTensorType>();
    RankedTensorType result_ty = op.getType().dyn_cast<RankedTensorType>();

    if (!lhs_ty || !rhs_ty || lhs_ty.getElementType() != f32_ty ||
        rhs_ty.getElementType() != f32_ty) {
      return failure();
    }

    Value lhs_f16 = rewriter.create<mhlo::ConvertOp>(loc, lhs, f16_ty);
    Value rhs_f16 = rewriter.create<mhlo::ConvertOp>(loc, rhs, f16_ty);
    RankedTensorType f16_tensor_ty =
        RankedTensorType::getChecked(loc, result_ty.getShape(), f16_ty);
    // tensor dot general
    Value dot = rewriter.create<mhlo::DotGeneralOp>(
        loc, f16_tensor_ty, lhs_f16, rhs_f16, op.getDotDimensionNumbers(),
        nullptr);
    Value fp32_dot = rewriter.create<mhlo::ConvertOp>(loc, dot, f32_ty);
    fp32_dot.setType(op.getResult().getType());
    rewriter.replaceOp(op, fp32_dot);
    return success();
  }
};

template <typename OpTy>
struct ElementwiseOpTypeConverter : public OpRewritePattern<OpTy> {
  ElementwiseOpTypeConverter(MLIRContext* context, Type from, Type to)
      : OpRewritePattern<OpTy>::OpRewritePattern(context) {
    this->from_ = from;
    this->to_ = to;
  }

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter& rewriter) const override {
    Location loc = op.getLoc();

    auto compareElemTy = [&](Type ty) {
      auto tensorTy = ty.dyn_cast<RankedTensorType>();
      if (!tensorTy) return false;
      return tensorTy.getElementType() == this->from_;
    };

    if (!llvm::all_of(op->getOperandTypes(), compareElemTy) ||
        !llvm::all_of(op->getResultTypes(), compareElemTy))
      return failure();

    SmallVector<Value> newOperands;
    for (Value operand : op->getOperands()) {
      auto rankedTy = operand.getType().cast<RankedTensorType>();
      auto newTy = RankedTensorType::get(rankedTy.getShape(), this->to_,
                                         rankedTy.getEncoding());
      newOperands.push_back(
          rewriter.create<mhlo::ConvertOp>(loc, newTy, operand));
    }
    SmallVector<Type> newTypes;
    for (Type ty : op->getResultTypes()) {
      auto rankedTy = ty.cast<RankedTensorType>();
      newTypes.push_back(RankedTensorType::get(rankedTy.getShape(), this->to_,
                                               rankedTy.getEncoding()));
    }
    auto newOp =
        rewriter.create<OpTy>(loc, newTypes, newOperands, op->getAttrs());
    SmallVector<Value> newResults;
    for (Value result : newOp->getResults()) {
      auto rankedTy = result.getType().cast<RankedTensorType>();
      auto newTy = RankedTensorType::get(rankedTy.getShape(), this->from_,
                                         rankedTy.getEncoding());
      newResults.push_back(
          rewriter.create<mhlo::ConvertOp>(loc, newTy, result));
    }
    rewriter.replaceOp(op, newResults);
    return success();
  }

  Type from_;
  Type to_;
};

struct ReduceOpTypeConverter : public OpRewritePattern<mhlo::ReduceOp> {
  ReduceOpTypeConverter(MLIRContext* context, Type from, Type to)
      : OpRewritePattern<mhlo::ReduceOp>::OpRewritePattern(context) {
    this->from_ = from;
    this->to_ = to;
  }

  LogicalResult matchAndRewrite(mhlo::ReduceOp op,
                                PatternRewriter& rewriter) const override {
    Location loc = op.getLoc();

    auto compareElemTy = [&](Type ty) {
      auto tensorTy = ty.dyn_cast<RankedTensorType>();
      if (!tensorTy) return false;
      return tensorTy.getElementType() == this->from_;
    };

    if (!llvm::all_of(op->getOperandTypes(), compareElemTy) ||
        !llvm::all_of(op->getResultTypes(), compareElemTy))
      return failure();

    SmallVector<Value> newOperands;
    for (Value operand : op->getOperands())
      newOperands.push_back(
          rewriter.create<mhlo::ConvertOp>(loc, operand, this->to_));

    mhlo::ReduceOp newOp =
        CloneAndReplaceElementType(op, rewriter, this->to_, newOperands);
    assert(newOp && "convert element type of reduce op failed");

    SmallVector<Value> newResults;
    for (Value result : newOp->getResults())
      newResults.push_back(
          rewriter.create<mhlo::ConvertOp>(loc, result, this->from_));
    rewriter.replaceOp(op, newResults);
    return success();
  }

  Type from_;
  Type to_;
};

struct ElementTypeConverterPass
    : public ElementTypeConverterPassBase<ElementTypeConverterPass> {
  explicit ElementTypeConverterPass(bool enable_fp16_gemm,
                                    bool enable_fp16_conv,
                                    bool promote_fp16_sensitive_ops_to_f32)
      : ElementTypeConverterPassBase<
            ElementTypeConverterPass>::ElementTypeConverterPassBase() {
    this->enable_fp16_gemm_ = enable_fp16_gemm;
    this->enable_fp16_conv_ = enable_fp16_conv;
    this->promote_fp16_sensitive_ops_to_f32_ =
        promote_fp16_sensitive_ops_to_f32;
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext& ctx = getContext();
    RewritePatternSet patterns(&ctx);
    patterns.insert<ConvertReduceOpWithSmallWidthIntType>(&ctx);
    if (enable_fp16_gemm_) {
      patterns.insert<ConvertDotGeneralOp>(&ctx);
    }
    if (enable_fp16_conv_) {
      patterns.insert<ConvertConvOp<mhlo::DynamicConvOp>,
                      ConvertConvOp<mhlo::ConvolutionOp>>(&ctx);
    }

    if (promote_fp16_sensitive_ops_to_f32_) {
      FloatType f16_ty = FloatType::getF16(&ctx);
      FloatType f32_ty = FloatType::getF32(&ctx);
      patterns.insert<ElementwiseOpTypeConverter<mhlo::RsqrtOp>>(&ctx, f16_ty,
                                                                 f32_ty);
      patterns.insert<ElementwiseOpTypeConverter<mhlo::TanhOp>>(&ctx, f16_ty,
                                                                f32_ty);
      patterns.insert<ReduceOpTypeConverter>(&ctx, f16_ty, f32_ty);
    }

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      func.emitError("applyPatternsAndFoldGreedily does not converge");
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createDiscElementTypeConverterPass(
    bool enable_fp16_gemm, bool enable_fp16_conv,
    bool promote_fp16_sensitive_ops_to_f32) {
  return std::make_unique<ElementTypeConverterPass>(
      enable_fp16_gemm, enable_fp16_conv, promote_fp16_sensitive_ops_to_f32);
}

}  // namespace disc_ral
}  // namespace mlir

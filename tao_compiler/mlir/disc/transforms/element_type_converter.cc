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

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"             // from @llvm-project
#include "mlir/IR/Attributes.h"                          // from @llvm-project
#include "mlir/IR/Location.h"                            // from @llvm-project
#include "mlir/IR/MLIRContext.h"                         // from @llvm-project
#include "mlir/IR/Operation.h"                           // from @llvm-project
#include "mlir/IR/PatternMatch.h"                        // from @llvm-project
#include "mlir/Pass/Pass.h"                              // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"                      // from @llvm-project
#include "transforms/PassDetail.h"

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
  block->addArguments({type, type});

  Location loc = body->getLoc();
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
  op.body().walk([&](Operation* hlo_op) {
    if (isa<mhlo::ReturnOp>(hlo_op)) return;
    ++num_non_return_hlo_ops;
    map_op = hlo_op;
  });

  assert(num_non_return_hlo_ops == 1 && map_op &&
         "Not support reduction region");

  Location loc = op.getLoc();
  mhlo::ReduceOp new_op = rewriter.create<mhlo::ReduceOp>(
      loc, operands[0], operands[1], op.dimensions());

  if (isa<mhlo::AddOp>(map_op)) {
    BuildReduceBody<mhlo::AddOp>(elem_type, &new_op.body(), &rewriter);
  } else if (isa<mhlo::MulOp>(map_op)) {
    BuildReduceBody<mhlo::MulOp>(elem_type, &new_op.body(), &rewriter);
  } else if (isa<mhlo::MaxOp>(map_op)) {
    BuildReduceBody<mhlo::MaxOp>(elem_type, &new_op.body(), &rewriter);
  } else if (isa<mhlo::MinOp>(map_op)) {
    BuildReduceBody<mhlo::MinOp>(elem_type, &new_op.body(), &rewriter);
  } else if (isa<mhlo::AndOp>(map_op)) {
    // We use MinOp to replace AndOp since std AndOp requires operands have i1
    // type.
    BuildReduceBody<mhlo::MinOp>(elem_type, &new_op.body(), &rewriter);
  } else if (isa<mhlo::OrOp>(map_op)) {
    // We use MaxOp to replace OrOp since std AndOp requires operands have i1
    // type.
    BuildReduceBody<mhlo::MaxOp>(elem_type, &new_op.body(), &rewriter);
  } else {
    assert(false && "not supported reduce type");
  }

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
    for (auto v : op.dimensions().getValues<APInt>()) {
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
    for (auto value : new_op.getResults()) {
      converted_results.push_back(
          rewriter.create<mhlo::ConvertOp>(loc, value, ty.getElementType()));
    }

    rewriter.replaceOp(op, converted_results);
    return success();
  }
};

struct ConvertDotGeneralOp : public OpRewritePattern<mhlo::DotGeneralOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::DotGeneralOp op,
                                PatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.lhs();
    Value rhs = op.rhs();
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
        loc, f16_tensor_ty, lhs_f16, rhs_f16, op.dot_dimension_numbers(),
        nullptr);
    Value fp32_dot = rewriter.create<mhlo::ConvertOp>(loc, dot, f32_ty);
    rewriter.replaceOp(op, fp32_dot);
    return success();
  }
};

struct ElementTypeConverterPass
    : public ElementTypeConverterPassBase<ElementTypeConverterPass> {
  explicit ElementTypeConverterPass(bool enable_fp16_gemm)
      : ElementTypeConverterPassBase<
            ElementTypeConverterPass>::ElementTypeConverterPassBase() {
    this->enable_fp16_gemm_ = enable_fp16_gemm;
  }

  void runOnFunction() override {
    auto func = getFunction();
    MLIRContext& ctx = getContext();
    OwningRewritePatternList patterns(&ctx);
    patterns.insert<ConvertReduceOpWithSmallWidthIntType>(&ctx);
    if (enable_fp16_gemm_) {
      patterns.insert<ConvertDotGeneralOp>(&ctx);
    }

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      func.emitError("applyPatternsAndFoldGreedily does not converge");
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createDiscElementTypeConverterPass(
    bool enable_fp16_gemm) {
  return std::make_unique<ElementTypeConverterPass>(enable_fp16_gemm);
}

}  // namespace disc_ral
}  // namespace mlir

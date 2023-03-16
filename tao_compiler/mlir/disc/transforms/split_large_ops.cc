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

// A pass that splits hlo ops that have too many operands into a sequence of hlo
// ops. This is because a GPU kernel supports limited number of parameters.

#include <iostream>

#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"                          // TF:llvm-project
#include "mlir/IR/Location.h"                            // TF:llvm-project
#include "mlir/IR/MLIRContext.h"                         // TF:llvm-project
#include "mlir/IR/Operation.h"                           // TF:llvm-project
#include "mlir/IR/PatternMatch.h"                        // TF:llvm-project
#include "mlir/Pass/Pass.h"                              // TF:llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // TF:llvm-project
#include "mlir/Transforms/Passes.h"                      // TF:llvm-project
#include "mlir/disc/transforms/PassDetail.h"

namespace mlir {
namespace disc_ral {

namespace {

struct ConvertConcatOp : public OpRewritePattern<mhlo::ConcatenateOp> {
  // using OpRewritePattern::OpRewritePattern;
  explicit ConvertConcatOp(MLIRContext* context, int max_num_operands_per_op)
      : OpRewritePattern<mhlo::ConcatenateOp>::OpRewritePattern(context) {
    this->max_num_operands_per_op_ = max_num_operands_per_op;
  }

  LogicalResult matchAndRewrite(mhlo::ConcatenateOp op,
                                PatternRewriter& rewriter) const override {
    int num_operands = op.getNumOperands();
    if (num_operands <= max_num_operands_per_op_) {
      return failure();
    }

    auto result_tp = op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!result_tp) {
      return failure();
    }

    auto loc = op.getLoc();
    SmallVector<Value, 4> lhs_operands;
    SmallVector<Value, 4> rhs_operands;
    for (int i = 0; i < num_operands; ++i) {
      Value operand = op.getOperand(i);
      if (i < num_operands / 2) {
        lhs_operands.push_back(operand);
      } else {
        rhs_operands.push_back(operand);
      }
    }

    auto rank = result_tp.getRank();
    SmallVector<int64_t, 4> sub_result_shape(rank, ShapedType::kDynamic);
    for (int64_t i = 0; i < rank; i++) {
      if (i != op.getDimension()) {
        sub_result_shape[i] = result_tp.getDimSize(i);
      }
    }
    auto sub_result_tp =
        RankedTensorType::get(sub_result_shape, result_tp.getElementType());
    Value lhs = rewriter.create<mhlo::ConcatenateOp>(
        loc, sub_result_tp, lhs_operands, op.getDimension());
    Value rhs = rewriter.create<mhlo::ConcatenateOp>(
        loc, sub_result_tp, rhs_operands, op.getDimension());

    SmallVector<Value, 2> fused_operands{lhs, rhs};
    auto fused_op = rewriter.create<mhlo::ConcatenateOp>(
        loc, result_tp, fused_operands, op.getDimension());

    rewriter.replaceOp(op, fused_op.getOperation()->getResults());
    return success();
  }

 private:
  int max_num_operands_per_op_;
};

struct SplitLargeOpsPass : public SplitLargeOpsPassBase<SplitLargeOpsPass> {
  explicit SplitLargeOpsPass(int max_num_operands_per_op)
      : SplitLargeOpsPassBase<SplitLargeOpsPass>::SplitLargeOpsPassBase() {
    this->max_num_operands_per_op_ = max_num_operands_per_op;
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(func.getContext());
    patterns.insert<ConvertConcatOp>(func.getContext(),
                                     max_num_operands_per_op_);

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      func.emitError("applyPatternsAndFoldGreedily does not converge");
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createDiscSplitLargeOpsPass(
    int max_num_operands_per_op) {
  return std::make_unique<SplitLargeOpsPass>(max_num_operands_per_op);
}

}  // namespace disc_ral
}  // namespace mlir

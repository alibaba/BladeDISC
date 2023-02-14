// Copyright 2021 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/disc/transforms/PassDetail.h"

namespace mlir {
namespace disc_ral {

namespace {
// Returns true if this op is a col or reduction.
bool isColOrRowReduction(Operation* op) {
  auto reduce_op = dyn_cast<mhlo::ReduceOp>(op);
  if (!reduce_op) return false;
  auto dimensions =
      llvm::to_vector<4>(reduce_op.getDimensions().getValues<int64_t>());
  llvm::sort(dimensions);
  auto ty = op->getOperand(0).getType().dyn_cast<RankedTensorType>();
  if (!ty) return false;
  bool isRowReduction = true;
  int maxDim = ty.getRank() - 1;
  for (int64_t dim : llvm::reverse(dimensions)) {
    if (dim != maxDim--) {
      isRowReduction = false;
      break;
    }
  }

  bool isColReduction = true;
  for (int64_t k = 0; k < dimensions.size(); k++) {
    if (k != dimensions[k]) {
      isColReduction = false;
      break;
    }
  }

  return isRowReduction or isColReduction;
}

struct ReduceOpConvert : public OpRewritePattern<mhlo::ReduceOp> {
  explicit ReduceOpConvert(MLIRContext* context) : OpRewritePattern(context) {}
  LogicalResult matchAndRewrite(mhlo::ReduceOp op,
                                PatternRewriter& rewriter) const override;
};

// Split two-side reduction into multiple reduction.
// For example, split the following reduction
//     acc = reduce(input, dimensions=[0, 2, 3])
// into:
//     acc0 = reduce(input, dimensions=[2, 3]) // row-reduction
//     acc = reduce(acc0, dimensions[0])  // col-reduction
LogicalResult ReduceOpConvert::matchAndRewrite(
    mhlo::ReduceOp op, PatternRewriter& rewriter) const {
  if (op.getInputs().size() != 1 || op.getInitValues().size() != 1 ||
      op.getResults().size() != 1)
    return failure();

  if (isColOrRowReduction(op)) return failure();
  auto input = op->getOperand(0);
  auto ty = input.getType().dyn_cast<RankedTensorType>();
  if (!ty) return failure();

  auto dimensions = llvm::to_vector<4>(op.getDimensions().getValues<int64_t>());
  llvm::sort(dimensions);
  auto inputRank = ty.getRank();
  int64_t d = 0;
  int64_t nReduceAxes = 0;
  // to find non-consecutive dimensions
  while (d + 1 < dimensions.size() && nReduceAxes == 0) {
    nReduceAxes = dimensions[d + 1] - dimensions[d] - 1;
    d++;
  }
  if (nReduceAxes + dimensions.size() != inputRank) return failure();

  // use lambda function to build mhlo::ReduceOp
  auto newReduce = [&](Value val, ArrayRef<int64_t> reduceDims) {
    auto newReduceOp =
        rewriter.create<mhlo::ReduceOp>(op.getLoc(), val, op.getOperand(1),
                                        rewriter.getI64TensorAttr(reduceDims));
    IRMapping mapping;
    mapping.clear();
    op.getBody().cloneInto(&newReduceOp.getBody(), mapping);
    return newReduceOp;
  };

  // build row-reduction
  auto rowDims =
      llvm::to_vector<4>(llvm::seq<int64_t>(d + nReduceAxes, inputRank));
  auto rowReduceOp = newReduce(op.getOperand(0), rowDims);

  // build col-reduction
  auto colDims = llvm::to_vector<4>(llvm::seq<int64_t>(0, d));
  auto colReduceOp = newReduce(rowReduceOp.getResult(0), colDims);
  rewriter.replaceOpWithNewOp<tensor::CastOp>(op, op.getType(0),
                                              colReduceOp.getResult(0));
  return success();
}
}  // namespace

struct ReductionRewriterPass
    : public ReductionRewriterPassBase<ReductionRewriterPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext* ctx = func.getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<ReduceOpConvert>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      func.emitError("applyPatternsAndFoldGreedily does not converge");
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<func::FuncOp>> createDiscReductionRewriterPass() {
  return std::make_unique<ReductionRewriterPass>();
}

}  // namespace disc_ral
}  // namespace mlir

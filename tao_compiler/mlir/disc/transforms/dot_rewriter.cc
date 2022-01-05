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

// This pass converts DotOp into DotGeneralOp, and folds transpose into
// DotGeneralOp.

#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

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

using llvm::StringRef;
using std::string;

namespace mlir {

namespace disc_ral {

namespace {

// It does not support transpose of batching dimension and other dimensions
// together now.
// TODO: it may support the transpose between batching dimensions only, or
// transpose of all batching dimensions together with the minor dimension.
static inline bool isNonBatchingTransposeTensorValue(
    Value val, SmallVector<int64_t, 4>& permutation,
    std::unordered_set<int64_t> batching_dims) {
  if (not val.getDefiningOp()) {
    return false;
  }
  permutation.clear();
  if (auto transpose = dyn_cast<mhlo::TransposeOp>(val.getDefiningOp())) {
    for (auto& en :
         llvm::enumerate(transpose.permutation().getValues<int64_t>())) {
      if (en.index() != en.value()) {
        if (batching_dims.find(en.index()) != batching_dims.end()) {
          return false;
        }
      }
      permutation.push_back(en.value());
    }
  }
  return !permutation.empty();
}

static inline DenseIntElementsAttr ConvertIntVecToDenseIntElementsAttr(
    llvm::ArrayRef<int64_t> op_dimensions, PatternRewriter& rewriter) {
  return DenseIntElementsAttr::get(
      RankedTensorType::get(op_dimensions.size(), rewriter.getIntegerType(64)),
      op_dimensions);
}

// Converts DotOp to DotGeneralOp.
struct DotToDotGeneralConvert : public OpRewritePattern<mhlo::DotOp> {
  explicit DotToDotGeneralConvert(MLIRContext* context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(mhlo::DotOp op,
                                PatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    Value old_lhs = op.lhs();
    Value old_rhs = op.rhs();

    std::vector<int64_t> lhs_contracting_dims;
    std::vector<int64_t> rhs_contracting_dims;
    // The operation performs sum of products over the second dimension of lhs
    // (or the first if it has rank 1) and the first dimension of rhs. These are
    // the "contracted" dimensions.
    // See https://www.tensorflow.org/xla/operation_semantics#dot
    lhs_contracting_dims.push_back(1);
    rhs_contracting_dims.push_back(0);
    auto dot_dimension_attr = mhlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(), {}, {}, lhs_contracting_dims,
        rhs_contracting_dims);

    Value dot_general = rewriter.create<mhlo::DotGeneralOp>(
        op.getLoc(), op.getType(), op.lhs(), op.rhs(), dot_dimension_attr,
        nullptr);
    rewriter.replaceOp(op, dot_general);

    return success();
  }
};

// Transpose folding into DotGeneralOp.
struct TransposeFoldingConvert : public OpRewritePattern<mhlo::DotGeneralOp> {
  explicit TransposeFoldingConvert(MLIRContext* context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(mhlo::DotGeneralOp op,
                                PatternRewriter& rewriter) const override {
    RankedTensorType result_ty = op.getType().dyn_cast<RankedTensorType>();
    if (!result_ty) {
      return failure();
    }

    Location loc = op.getLoc();
    Value old_lhs = op.lhs();
    Value old_rhs = op.rhs();

    RankedTensorType old_l_type =
        old_lhs.getType().dyn_cast<RankedTensorType>();
    RankedTensorType old_r_type =
        old_rhs.getType().dyn_cast<RankedTensorType>();
    if ((!old_l_type || !old_r_type)) {
      return failure();
    }

    auto dim_numbers = op.dot_dimension_numbers();
    auto lhs_batching_dims = dim_numbers.getLhsBatchingDimensions();
    SmallVector<int64_t, 4> lhs_perm;
    bool tp_lhs = isNonBatchingTransposeTensorValue(
        old_lhs, lhs_perm,
        std::unordered_set<int64_t>(lhs_batching_dims.begin(),
                                    lhs_batching_dims.end()));
    SmallVector<int64_t, 4> rhs_perm;

    auto rhs_batching_dims = dim_numbers.getRhsBatchingDimensions();
    bool tp_rhs = isNonBatchingTransposeTensorValue(
        old_rhs, rhs_perm,
        std::unordered_set<int64_t>(rhs_batching_dims.begin(),
                                    rhs_batching_dims.end()));

    if (!tp_lhs && !tp_rhs) {
      return failure();
    }

    std::vector<int64_t> lhs_contracting_dims;
    if (tp_lhs) {
      for (auto& en :
           llvm::enumerate(dim_numbers.getLhsContractingDimensions())) {
        lhs_contracting_dims.push_back(lhs_perm[en.value()]);
      }
    } else {
      lhs_contracting_dims = dim_numbers.getLhsContractingDimensions();
    }

    std::vector<int64_t> rhs_contracting_dims;
    if (tp_rhs) {
      for (auto& en :
           llvm::enumerate(dim_numbers.getRhsContractingDimensions())) {
        rhs_contracting_dims.push_back(rhs_perm[en.value()]);
      }
    } else {
      rhs_contracting_dims = dim_numbers.getRhsContractingDimensions();
    }

    auto dot_dimension_attr = mhlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(), dim_numbers.getLhsBatchingDimensions(),
        dim_numbers.getRhsBatchingDimensions(), lhs_contracting_dims,
        rhs_contracting_dims);

    // Re-direct the lhs/rhs if needed.
    Value lhs = tp_lhs ? old_lhs.getDefiningOp()->getOperand(0) : old_lhs;
    Value rhs = tp_rhs ? old_rhs.getDefiningOp()->getOperand(0) : old_rhs;

    Value dot = rewriter.create<mhlo::DotGeneralOp>(
        loc, op.getType(), lhs, rhs, dot_dimension_attr, nullptr);
    rewriter.replaceOp(op, dot);

    // Remove transpose op which outputs into dot.
    if (tp_lhs) {
      rewriter.eraseOp(old_lhs.getDefiningOp());
    }
    if (tp_rhs) {
      rewriter.eraseOp(old_rhs.getDefiningOp());
    }

    return success();
  }
};

struct DotRewriterPass : public DotRewriterPassBase<DotRewriterPass> {
  void runOnFunction() override {
    FuncOp func = getFunction();
    // TODO: if needs to do const reformat, we need the xla_hlo.dot with its
    // inputs

    MLIRContext* ctx = func.getContext();
    OwningRewritePatternList patterns(ctx);
    patterns.insert<DotToDotGeneralConvert, TransposeFoldingConvert>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      func.emitError("applyPatternsAndFoldGreedily does not converge");
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createDiscDotRewriterPass() {
  return std::make_unique<DotRewriterPass>();
}

}  // namespace disc_ral
}  // namespace mlir

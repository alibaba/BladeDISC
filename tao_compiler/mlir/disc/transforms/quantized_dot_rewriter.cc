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
#include <unordered_set>

#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"                          // TF:llvm-project
#include "mlir/IR/Operation.h"                           // TF:llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"                      // TF:llvm-project
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/placement_utils.h"

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
         llvm::enumerate(transpose.getPermutation().getValues<int64_t>())) {
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

RankedTensorType GetTransposeOutputType(
    Value value, const SmallVectorImpl<int64_t>& transpose_permutation,
    OpBuilder& b) {
  // Compute the resulting shape.
  llvm::SmallVector<int64_t, 4> transposed_shape;
  ShapedType input_type = value.getType().cast<ShapedType>();
  auto input_shape = input_type.getShape();
  for (int64_t val : transpose_permutation) {
    transposed_shape.push_back(input_shape[val]);
  }
  return RankedTensorType::get(transposed_shape, input_type.getElementType());
}

Operation* InsertTranspose(
    mlir::mhlo_disc::QuantizedDotGeneralOp op, Value value,
    const SmallVectorImpl<int64_t>& transpose_permutation, OpBuilder& b) {
  auto transpose_permutation_attr =
      GetI64ElementsAttr(transpose_permutation, &b);

  auto transpose_type = GetTransposeOutputType(value, transpose_permutation, b);
  auto transpose_op = b.create<mhlo::TransposeOp>(
      op.getLoc(), transpose_type, value, transpose_permutation_attr);

  if (auto attr = op->getAttr(placement_utils::kDiscPlaceAssignment))
    transpose_op->setAttr(placement_utils::kDiscPlaceAssignment, attr);

  return transpose_op;
}

// The basic logic of this pass is to insert transpose op to ensure the qgemm op
// having expected format.
struct QuantDotTransposeConvert
    : public OpRewritePattern<mlir::mhlo_disc::QuantizedDotGeneralOp> {
  explicit QuantDotTransposeConvert(MLIRContext* context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(mlir::mhlo_disc::QuantizedDotGeneralOp op,
                                PatternRewriter& rewriter) const override {
    bool onGpu = placement_utils::isGpuMhlo(op);
    // Only need to transpose weight on gpu currently
    if (onGpu == false) {
      return failure();
    }

    auto inputTy = op.getInput().getType().dyn_cast<RankedTensorType>();
    auto weightTy = op.getWeight().getType().dyn_cast<RankedTensorType>();

    // This transpose pass can only support dot with rank equal to 2
    if (inputTy.getRank() != 2 || weightTy.getRank() != 2) {
      return failure();
    }

    // Match fail if weight is nxk
    std::vector<int64_t> rhs_contracting_dims_check;
    auto dim_numbers = op.getDotDimensionNumbers();
    rhs_contracting_dims_check = dim_numbers.getRhsContractingDimensions();
    // Only match matrix whose contracting_dims.size() is 1
    if (rhs_contracting_dims_check.size() != 1) {
      return failure();
    }

    if (rhs_contracting_dims_check[0] == 1) {
      return failure();
    }

    SmallVector<int64_t> transposeAttr = {1, 0};

    OpBuilder b(op);
    auto transpose_op = InsertTranspose(op, op.getWeight(), transposeAttr, b);

    Location loc = op.getLoc();
    Value old_lhs = op.getInput();
    Value old_rhs = transpose_op->getResult(0);

    RankedTensorType old_l_type =
        old_lhs.getType().dyn_cast<RankedTensorType>();
    RankedTensorType old_r_type =
        old_rhs.getType().dyn_cast<RankedTensorType>();
    if ((!old_l_type || !old_r_type)) {
      return failure();
    }

    SmallVector<int64_t, 4> lhs_perm;
    auto lhs_batching_dims = dim_numbers.getLhsBatchingDimensions();
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

    Value newQuantizedDot =
        rewriter.create<mlir::mhlo_disc::QuantizedDotGeneralOp>(
            loc, op.getType(), op.getInput(), transpose_op->getResult(0),
            op.getInputScale(), op.getInputZeroPoint(), op.getWeightScale(),
            op.getWeightZeroPoint(), op.getResultScale(),
            op.getResultZeroPoint(), dot_dimension_attr, op.getUseSymmetric(),
            op.getAxis(), op.getUseDynamic());

    rewriter.replaceOp(op, newQuantizedDot);
    return success();
  }
};

struct DiscQuantizedDotRewriterPass
    : public QuantizedDotRewriterPassBase<DiscQuantizedDotRewriterPass> {
  explicit DiscQuantizedDotRewriterPass()
      : QuantizedDotRewriterPassBase<
            DiscQuantizedDotRewriterPass>::QuantizedDotRewriterPassBase() {}

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    MLIRContext* ctx = func.getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<QuantDotTransposeConvert>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      func.emitError("applyPatternsAndFoldGreedily does not converge");
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createDiscQuantizedDotRewriter() {
  return std::make_unique<DiscQuantizedDotRewriterPass>();
}

}  // namespace disc_ral
}  // namespace mlir

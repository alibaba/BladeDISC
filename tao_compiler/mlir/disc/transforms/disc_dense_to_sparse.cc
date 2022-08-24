// Copyright 2022 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iostream"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/disc/IR/hlo_disc_ops.h"
#include "tensorflow/compiler/mlir/disc/disc_util.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"
#include "tensorflow/compiler/mlir/disc/transforms/disc_shape_optimization_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/placement_utils.h"

#define DEBUG_TYPE "disc-dense-to-sparse"

// This file implements some basic optimizations to do algebra simplification.

namespace mlir {
namespace disc_ral {
namespace {

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
    mhlo::DotGeneralOp op, Value value,
    const SmallVectorImpl<int64_t>& transpose_permutation, OpBuilder& b) {
  auto transpose_permutation_attr =
      GetI64ElementsAttr(transpose_permutation, &b);

  // Compute the resulting shape.
  llvm::SmallVector<int64_t, 4> transposed_shape;
  ShapedType input_type = value.getType().cast<ShapedType>();
  auto input_shape = input_type.getShape();
  for (auto val : transpose_permutation) {
    transposed_shape.push_back(input_shape[val]);
  }
  auto transpose_type = GetTransposeOutputType(value, transpose_permutation, b);
  auto transpose_op = b.create<mhlo::TransposeOp>(
      op.getLoc(), transpose_type, value, transpose_permutation_attr);

  if (auto attr = op->getAttr(placement_utils::kDiscPlaceAssignment))
    transpose_op->setAttr(placement_utils::kDiscPlaceAssignment, attr);

  return transpose_op;
}

// convert:
//   mhlo.dense_gemm(x, w)
// to:
//   mhlo.transpose(x, [1, 0])
//   mhlo.sparse_gemm(x, w)
//   mhlo.transpose(y, [1, 0])
struct ExpandDotGeneralOp : public OpRewritePattern<mhlo::DotGeneralOp> {
  using OpRewritePattern<mhlo::DotGeneralOp>::OpRewritePattern;

  LogicalResult SparseOp(mhlo::DotGeneralOp op, PatternRewriter& rewriter,
                         int sparse_weight_pos) const {
    OpBuilder b(op);
    Location loc = op.getLoc();

    Value input, kernel;
    if (sparse_weight_pos == 0) {
      input = op->getOperand(1);
      kernel = op->getOperand(0);
    } else {
      input = op->getOperand(0);
      kernel = op->getOperand(1);
    }
    Value output = op->getResult(0);

    auto inputTy = input.getType().dyn_cast<RankedTensorType>();
    auto kernelTy = kernel.getType().dyn_cast<RankedTensorType>();
    auto outputTy = output.getType().dyn_cast<RankedTensorType>();

    if (!inputTy || !kernelTy || !outputTy) return success();
    if (inputTy.getShape().size() != 2) return success();

    SmallVector<int64_t> transposeAttr = {1, 0};

    SmallVector<Value> newOperands;

    // insert transpose before spgemm when sparse_weight_pos = 1
    if (sparse_weight_pos == 1) {
      auto inputTranOp = InsertTranspose(op, input, transposeAttr, b);
      newOperands.push_back(inputTranOp->getResult(0));
    } else {
      newOperands.push_back(input);
    }

    newOperands.push_back(kernel);

    // compute spgemm output shape
    auto output_shape = outputTy.getShape();
    llvm::SmallVector<int64_t, 4> spgemm_shape;
    if (sparse_weight_pos == 1) {
      for (int64_t val : transposeAttr) {
        spgemm_shape.push_back(output_shape[val]);
      }
    } else {
      for (int64_t val : output_shape) {
        spgemm_shape.push_back(val);
      }
    }

    auto newOutputTy =
        RankedTensorType::get(spgemm_shape, outputTy.getElementType());

    // insert spgemm op
    auto dot_dimension_numbers = op.dot_dimension_numbers();
    auto lhs_contracting_dimensions =
        dot_dimension_numbers.getLhsContractingDimensions();
    auto rhs_contracting_dimensions =
        dot_dimension_numbers.getRhsContractingDimensions();

    int64_t lhs_cd, rhs_cd;
    if (sparse_weight_pos == 1) {
      lhs_cd = 1 - lhs_contracting_dimensions[0];
      rhs_cd = rhs_contracting_dimensions[0];
    } else {
      lhs_cd = rhs_contracting_dimensions[0];
      rhs_cd = lhs_contracting_dimensions[0];
    }

    auto lhs_cd_attr = b.getNamedAttr("lhs_contracting_dimensions",
                                      b.getI64IntegerAttr(lhs_cd));
    auto rhs_cd_attr = b.getNamedAttr("rhs_contracting_dimensions",
                                      b.getI64IntegerAttr(rhs_cd));

    NamedAttribute attrs[] = {lhs_cd_attr, rhs_cd_attr};
    auto config = b.getDictionaryAttr(attrs);

    auto spgemm = b.create<mhlo_disc::CustomCallOp>(
        loc, newOutputTy, newOperands, "sparse_gemm", false, config);

    spgemm.getResult(0).setType(newOutputTy);

    // insert transpose after spgemm when sparse_weight_pos = 1
    if (sparse_weight_pos == 1) {
      auto output_transpose_op =
          InsertTranspose(op, spgemm.getResult(0), transposeAttr, b);
      output_transpose_op->getResult(0).setType(outputTy);
      rewriter.replaceOp(op, output_transpose_op->getResult(0));
    } else {
      rewriter.replaceOp(op, spgemm.getResult(0));
    }
    return success();
  }

  LogicalResult extractWeightFromConst(mhlo::ConstantOp constOp,
                                       bool is_transpose) const {
    if (!constOp) return failure();

    auto constTy = constOp.getResult().getType().cast<RankedTensorType>();
    if (!constTy.getElementType().isF16()) return failure();

    auto weight_type = constOp.getType().cast<ShapedType>();
    ArrayRef<int64_t> weight_shape = weight_type.getShape();

    if (weight_shape.size() != 2) return failure();

    auto weight_value = constOp.value().getValues<APFloat>().begin();

    int64_t row = is_transpose ? weight_shape[1] : weight_shape[0];
    int64_t col = is_transpose ? weight_shape[0] : weight_shape[1];

    // check sparse
    if (row == 0 || col == 0 || col % 4 != 0) {
      return failure();
    }

    for (int i = 0; i < row; ++i) {
      for (int j = 0; j < col / 4; ++j) {
        int zero_count = 0;
        for (int k = 0; k < 4; ++k) {
          if (is_transpose) {
            if ((*(weight_value + (j * 4 + k) * row + i)).convertToDouble() ==
                0)
              zero_count++;
          } else {
            if ((*(weight_value + i * col + j * 4 + k)).convertToDouble() == 0)
              zero_count++;
          }
        }
        if (zero_count < 2) {
          return failure();
        }
      }
    }
    return success();
  }

  int tryToFindSparseWeight(mhlo::DotGeneralOp op) const {
    Operation* rhsDefiningOp = op.rhs().getDefiningOp();
    if (!rhsDefiningOp) return -1;

    if (auto constOp = dyn_cast<mhlo::ConstantOp>(rhsDefiningOp)) {
      if (succeeded(extractWeightFromConst(constOp, true))) {
        return 1;
      }
    }

    Operation* lhsDefiningOp = op.lhs().getDefiningOp();
    if (!lhsDefiningOp) return -1;

    if (auto constOp = dyn_cast<mhlo::ConstantOp>(lhsDefiningOp)) {
      if (succeeded(extractWeightFromConst(constOp, false))) {
        return 0;
      }
    }

    return -1;
  }

  LogicalResult matchAndRewrite(mhlo::DotGeneralOp op,
                                PatternRewriter& rewriter) const override {
    int sparse_weight_pos = tryToFindSparseWeight(op);
    if (sparse_weight_pos == -1) return failure();

    return SparseOp(op, rewriter, sparse_weight_pos);
  }
};

void populateDiscDenseToSparsePatterns(RewritePatternSet& patterns) {
  // clang-format off
  patterns.insert<ExpandDotGeneralOp>(patterns.getContext());
  // clang-format on
}

struct DiscDenseToSparsePass
    : public DiscDenseToSparsePassBase<DiscDenseToSparsePass> {
  explicit DiscDenseToSparsePass()
      : DiscDenseToSparsePassBase<
            DiscDenseToSparsePass>::DiscDenseToSparsePassBase() {}
  void runOnOperation() override;
};

void DiscDenseToSparsePass::runOnOperation() {
  // Setup rewriter patterns.
  MLIRContext& ctx = getContext();
  RewritePatternSet patterns(&ctx);
  populateDiscDenseToSparsePatterns(patterns);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
    return;
  }
}

// ----------------------------------------------------------------------

// convert:
//   mhlo.transpose(x, [1, 0])
//   mhlo.sparse_gemm(x, w)
// to:
//   mhlo.sparse_gemm(x, w)
struct ExpandSparseGemmOp : public OpRewritePattern<mhlo_disc::CustomCallOp> {
  using OpRewritePattern<mhlo_disc::CustomCallOp>::OpRewritePattern;

  LogicalResult NewSparseOp(mhlo_disc::CustomCallOp op,
                            PatternRewriter& rewriter) const {
    OpBuilder b(op);
    Location loc = op.getLoc();

    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    Value output = op->getResult(0);

    auto inputTy = lhs.getType().dyn_cast<RankedTensorType>();
    auto kernelTy = rhs.getType().dyn_cast<RankedTensorType>();
    auto outputTy = output.getType().dyn_cast<RankedTensorType>();

    if (!inputTy || !kernelTy || !outputTy) return success();

    auto config = op.backend_config().cast<DictionaryAttr>();
    int64_t lhs_contracting_dimensions =
        config.getAs<IntegerAttr>("lhs_contracting_dimensions").getInt();
    int64_t rhs_contracting_dimensions =
        config.getAs<IntegerAttr>("rhs_contracting_dimensions").getInt();

    SmallVector<Value> newOperands;

    // check lhs
    Operation* lhsDefiningOp = lhs.getDefiningOp();
    bool lhsTrans = false;
    if (lhsDefiningOp && dyn_cast<mhlo::TransposeOp>(lhsDefiningOp)) {
      lhs_contracting_dimensions = 1 - lhs_contracting_dimensions;
      newOperands.push_back(lhsDefiningOp->getOperand(0));
    } else {
      newOperands.push_back(lhs);
    }

    // check rhs
    Operation* rhsDefiningOp = rhs.getDefiningOp();
    bool rhsTrans = false;
    if (rhsDefiningOp && dyn_cast<mhlo::TransposeOp>(rhsDefiningOp)) {
      rhs_contracting_dimensions = 1 - rhs_contracting_dimensions;
      newOperands.push_back(rhsDefiningOp->getOperand(0));
    } else {
      newOperands.push_back(rhs);
    }

    auto lhsConDim =
        b.getNamedAttr("lhs_contracting_dimensions",
                       b.getI64IntegerAttr(lhs_contracting_dimensions));
    auto rhsConDim =
        b.getNamedAttr("rhs_contracting_dimensions",
                       b.getI64IntegerAttr(rhs_contracting_dimensions));

    NamedAttribute attrs[] = {lhsConDim, rhsConDim};
    auto newConfig = b.getDictionaryAttr(attrs);

    auto spgemm = b.create<mhlo_disc::CustomCallOp>(
        loc, outputTy, newOperands, "sparse_gemm", false, newConfig);
    spgemm.getResult(0).setType(outputTy);
    rewriter.replaceOp(op, spgemm->getResult(0));

    return success();
  }

  LogicalResult tryToFindSparseGemm(mhlo_disc::CustomCallOp op) const {
    auto call_target_name = op.call_target_name();
    if (call_target_name == "sparse_gemm") {
      Value lhs = op->getOperand(0);
      Value rhs = op->getOperand(1);
      Operation* lhsDefiningOp = lhs.getDefiningOp();
      if (!lhsDefiningOp) return failure();

      if (auto transOp = dyn_cast<mhlo::TransposeOp>(lhsDefiningOp)) {
        return success();
      }

      Operation* rhsDefiningOp = rhs.getDefiningOp();
      if (!rhsDefiningOp) return failure();

      if (auto transOp = dyn_cast<mhlo::TransposeOp>(rhsDefiningOp)) {
        return success();
      }
    }
    return failure();
  }

  LogicalResult matchAndRewrite(mhlo_disc::CustomCallOp op,
                                PatternRewriter& rewriter) const override {
    if (failed(tryToFindSparseGemm(op))) return failure();

    return NewSparseOp(op, rewriter);
  }
};

void populateDiscSparseGemmTransposeSimplifierPatterns(
    RewritePatternSet& patterns) {
  // clang-format off
  patterns.insert<
    ExpandSparseGemmOp
  >(patterns.getContext());
  // clang-format on
}

struct DiscSparseGemmTransposeSimplifierPass
    : public DiscSparseGemmTransposeSimplifierPassBase<
          DiscSparseGemmTransposeSimplifierPass> {
  explicit DiscSparseGemmTransposeSimplifierPass()
      : DiscSparseGemmTransposeSimplifierPassBase<
            DiscSparseGemmTransposeSimplifierPass>::
            DiscSparseGemmTransposeSimplifierPassBase() {}
  void runOnOperation() override;
};

void DiscSparseGemmTransposeSimplifierPass::runOnOperation() {
  // Setup rewriter patterns.
  MLIRContext& ctx = getContext();
  RewritePatternSet patterns(&ctx);
  populateDiscSparseGemmTransposeSimplifierPatterns(patterns);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
    return;
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createDiscDenseToSparsePass() {
  return std::make_unique<DiscDenseToSparsePass>();
}

std::unique_ptr<OperationPass<func::FuncOp>>
createDiscSparseGemmTransposeSimplifierPass() {
  return std::make_unique<DiscSparseGemmTransposeSimplifierPass>();
}

}  // namespace disc_ral
}  // namespace mlir

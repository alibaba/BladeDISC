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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/disc/IR/hlo_disc_ops.h"
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/disc_shape_optimization_utils.h"
#include "mlir/disc/transforms/placement_utils.h"

#define DEBUG_TYPE "disc-dense-to-sparse"

// This file implements some basic optimizations to do algebra simplification.

namespace mlir {
namespace disc_ral {
namespace {

template <typename MetaDtype>
bool generate_sparse_balance_weight(bool w_transpose, int64_t w_dim0,
                                    int64_t w_dim1,
                                    detail::ElementsAttrIterator<APFloat> ptr_W,
                                    APFloat* ptr_compress_W, MetaDtype* ptr_M,
                                    MetaDtype* ptr_M_buf) {
  // float16
  int step = 4;
  int m = 2;
  int n = 4;
  int dtype_size = 2;

  int sparse_element = 2;
  int element_per_m = 16 / std::log2(n);
  int decom_element_per_m = 32 / dtype_size;
  int row = w_dim0;
  int col = w_dim1;

  for (int r = 0; r < row; ++r) {
    int w_count = 0;
    for (int c = 0; c < (col / decom_element_per_m); ++c) {
      std::vector<int> unremove_indices;
      for (int i = 0; i < decom_element_per_m; i++) {
        long long int a_index = 0;
        if (w_transpose) {
          a_index = (c * decom_element_per_m + i) * row + r;
        } else {
          a_index = r * col + c * decom_element_per_m + i;
        }
        if ((*(ptr_W + a_index)).convertToFloat() != 0) {
          if (w_transpose) {
            ptr_compress_W[w_count * row + r] = *(ptr_W + a_index);
          } else {
            ptr_compress_W[r * col / sparse_element + w_count] =
                *(ptr_W + a_index);
          }
          unremove_indices.push_back(i % n);
          w_count++;
        }
      }
      int e_indices = r * col / decom_element_per_m + c;
      ptr_M_buf[e_indices] = 0;
      for (int i = 0; i < unremove_indices.size(); ++i) {
        ptr_M_buf[e_indices] |= (unremove_indices[i] << (2 * i));
      }
    }
  }

  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col / sparse_element / element_per_m; j++) {
      int group = (sizeof(MetaDtype) == 2) ? 32 : 16;
      int interweave = (sizeof(MetaDtype) == 2) ? 4 : 2;

      int dest_row = i / group * group + (i % 8) * interweave + (i % group) / 8;
      int dest_col = j;

      if (((dest_row % 2) == 0) && ((dest_col % 2) == 1)) {
        ++dest_row;
        --dest_col;
      } else if (((dest_row % 2) == 1) && ((dest_col % 2) == 0)) {
        --dest_row;
        ++dest_col;
      }

      int dest_col_major = dest_col / 2;
      int dest_col_minor = dest_col % 2;

      ptr_M[dest_col_major * row * 2 + dest_row * 2 + dest_col_minor] =
          ptr_M_buf[i * col / sparse_element / element_per_m + j];
    }
  }

  return true;
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
// enable_sparse_convert = False
//   mhlo.transpose(x, [1, 0])
//   mhlo.sparse_gemm(x, w)
//   mhlo.transpose(y, [1, 0])
// enable_sparse_convert = True
//   mhlo.transpose(x, [1, 0])
//   mhlo.sparse_gemm(x, sp_w_data, sp_w_indices)
//   mhlo.transpose(y, [1, 0])
struct ExpandDotGeneralOp : public OpRewritePattern<mhlo::DotGeneralOp> {
  using OpRewritePattern<mhlo::DotGeneralOp>::OpRewritePattern;

  bool enable_sparse_convert_ = false;

  ExpandDotGeneralOp(MLIRContext* context, bool enable_sparse_convert,
                     PatternBenefit benefit = 1,
                     ArrayRef<StringRef> generatedNames = {})
      : OpRewritePattern<mhlo::DotGeneralOp>(context, benefit, generatedNames) {
    enable_sparse_convert_ = enable_sparse_convert;
  }

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

    // convert dense weight to sparse weight
    if (enable_sparse_convert_) {
      Operation* weightOp;
      if (sparse_weight_pos == 0) {
        weightOp = op.getLhs().getDefiningOp();
      } else {
        weightOp = op.getRhs().getDefiningOp();
      }
      auto weightConstOp = dyn_cast<mhlo::ConstantOp>(weightOp);
      ArrayRef<int64_t> weightShape =
          weightConstOp.getType().cast<ShapedType>().getShape();

      auto weights_value =
          weightConstOp.getValue().getValues<APFloat>().begin();
      SmallVector<APFloat> compressed_weight(
          weightShape[0] * weightShape[1] / 2, APFloat(0.f));
      std::vector<uint16_t> compressed_index(
          weightShape[0] * weightShape[1] / 16, 0);
      std::vector<uint16_t> compressed_reorder_index(
          weightShape[0] * weightShape[1] / 16, 0);
      int weight_row = sparse_weight_pos == 0 ? weightShape[0] : weightShape[1];
      int weight_col = sparse_weight_pos == 0 ? weightShape[1] : weightShape[0];
      generate_sparse_balance_weight<uint16_t>(
          sparse_weight_pos == 1, weight_row, weight_col, weights_value,
          compressed_weight.data(), compressed_reorder_index.data(),
          compressed_index.data());

      SmallVector<int64_t> spWeightShape;
      if (sparse_weight_pos == 0) {
        spWeightShape.push_back(weightShape[0]);
        spWeightShape.push_back(weightShape[1] / 2);
      } else {
        spWeightShape.push_back(weightShape[0] / 2);
        spWeightShape.push_back(weightShape[1]);
      }

      auto spWeightDataDtype =
          RankedTensorType::get(spWeightShape, kernelTy.getElementType());
      auto spWeightIndexDtype = RankedTensorType::get(
          {weightShape[0] * weightShape[1] / 16}, b.getIntegerType(16, false));

      auto spWeightDataOp = b.create<mhlo::ConstantOp>(
          op->getLoc(), spWeightDataDtype,
          DenseElementsAttr::get(spWeightDataDtype,
                                 llvm::makeArrayRef(compressed_weight)));

      auto spWeightIndexOp = b.create<mhlo::ConstantOp>(
          op->getLoc(), spWeightIndexDtype,
          DenseElementsAttr::get(spWeightIndexDtype,
                                 llvm::makeArrayRef(compressed_reorder_index)));

      newOperands.push_back(spWeightDataOp->getResult(0));
      newOperands.push_back(spWeightIndexOp->getResult(0));
    } else {
      newOperands.push_back(kernel);
    }

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
    auto dot_dimension_numbers = op.getDotDimensionNumbers();
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

    auto weight_value = constOp.getValue().getValues<APFloat>().begin();

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
    Operation* rhsDefiningOp = op.getRhs().getDefiningOp();
    if (!rhsDefiningOp) return -1;

    if (auto constOp = dyn_cast<mhlo::ConstantOp>(rhsDefiningOp)) {
      if (succeeded(extractWeightFromConst(constOp, true))) {
        return 1;
      }
    }

    Operation* lhsDefiningOp = op.getLhs().getDefiningOp();
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

void populateDiscDenseToSparsePatterns(RewritePatternSet& patterns,
                                       bool enable_sparse_convert) {
  // clang-format off
  patterns.insert<ExpandDotGeneralOp>(patterns.getContext(), enable_sparse_convert);
  // clang-format on
}

struct DiscDenseToSparsePass
    : public DiscDenseToSparsePassBase<DiscDenseToSparsePass> {
  explicit DiscDenseToSparsePass(bool enable_sparse_convert)
      : DiscDenseToSparsePassBase<
            DiscDenseToSparsePass>::DiscDenseToSparsePassBase() {
    this->enable_sparse_convert_ = enable_sparse_convert;
  }
  void runOnOperation() override;
};

void DiscDenseToSparsePass::runOnOperation() {
  // Setup rewriter patterns.
  MLIRContext& ctx = getContext();
  RewritePatternSet patterns(&ctx);
  populateDiscDenseToSparsePatterns(patterns, enable_sparse_convert_);
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

    // first input of sparse_gemm definitely is input_data
    Value lhs = op->getOperand(0);
    Value output = op->getResult(0);

    auto inputTy = lhs.getType().dyn_cast<RankedTensorType>();
    auto outputTy = output.getType().dyn_cast<RankedTensorType>();

    if (!inputTy || !outputTy) return success();

    auto config = op.getBackendConfig().cast<DictionaryAttr>();
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

    for (int i = 1; i < op->getNumOperands(); ++i) {
      newOperands.push_back(op->getOperand(i));
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
    auto call_target_name = op.getCallTargetName();
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

std::unique_ptr<OperationPass<func::FuncOp>> createDiscDenseToSparsePass(
    bool enable_sparse_convert) {
  return std::make_unique<DiscDenseToSparsePass>(enable_sparse_convert);
}

std::unique_ptr<OperationPass<func::FuncOp>>
createDiscSparseGemmTransposeSimplifierPass() {
  return std::make_unique<DiscSparseGemmTransposeSimplifierPass>();
}

}  // namespace disc_ral
}  // namespace mlir

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

// This pass converts DotOp into DotGeneralOp, folds transpose into
// DotGeneralOp, and do necessary layout legalization for DotGeneralOp

#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"       // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"                // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"               // from @llvm-project
#include "mlir/IR/Attributes.h"                          // from @llvm-project
#include "mlir/IR/Location.h"                            // from @llvm-project
#include "mlir/IR/MLIRContext.h"                         // from @llvm-project
#include "mlir/IR/Operation.h"                           // from @llvm-project
#include "mlir/IR/PatternMatch.h"                        // from @llvm-project
#include "mlir/Pass/Pass.h"                              // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"                      // from @llvm-project
#include "tensorflow/compiler/mlir/disc/disc_util.h"
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

// Lower EinsumOp to DotGeneralOp with possible layout adjustment.
// A DotGeneralOp is acceptable only if:
// 1, batching dimensions must be in lower dimensions
// 2, batching dimensions of lhs/rhs/result must be the same
// 3, one contracing dimension for lhs/rhs, which must be among the
//    last two dimensions of is acceptable in case
//
// step 1, analysis of equation string to get contracting/batching token
// step 2, transpose all the batching dims to the lower dimensions, and
//         all the non_contracting dimensions to be adjacent
// step 3, reshape if more than one non-contracting/non-batching dims
//         for lhs/rhs/result
struct EinsumToDotGeneralPattern : public OpRewritePattern<mhlo::EinsumOp> {
  using TokensTy =
      llvm::SmallDenseMap<char, llvm::SmallDenseMap<EquationVariable, size_t>>;
  explicit EinsumToDotGeneralPattern(MLIRContext* context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(mhlo::EinsumOp op,
                                PatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    Value old_lhs = op.lhs();
    Value old_rhs = op.rhs();
    StringRef equation = op.einsum_config();

    TokensTy tokens;
    llvm::SmallDenseSet<char> contracting_tokens;
    llvm::SmallDenseSet<char> batching_tokens;
    llvm::SmallDenseSet<char> lhs_non_contracting_tokens;
    llvm::SmallDenseSet<char> rhs_non_contracting_tokens;

    SmallVector<char> lhs_original_tokens;
    SmallVector<char> rhs_original_tokens;
    SmallVector<char> result_original_tokens;

    bool isLhsBNC;
    bool isRhsBNC;

    SmallVector<char> lhs_reordered_tokens;
    SmallVector<char> rhs_reordered_tokens;
    SmallVector<char> result_reordered_tokens;

    if (!parseEinsumEquation(equation, tokens, &lhs_original_tokens,
                             &rhs_original_tokens, &result_original_tokens)) {
      return op.emitError("unexpected equation") << equation << "\n";
    }

    categorizeTokens(tokens, contracting_tokens, batching_tokens,
                     lhs_non_contracting_tokens, rhs_non_contracting_tokens);

    getReorderedTokens(
        contracting_tokens, batching_tokens, lhs_non_contracting_tokens,
        rhs_non_contracting_tokens, lhs_original_tokens, rhs_original_tokens,
        result_original_tokens, isLhsBNC, isRhsBNC, lhs_reordered_tokens,
        rhs_reordered_tokens, result_reordered_tokens);

    Value lhs = processOperand(rewriter, old_lhs, loc, lhs_original_tokens,
                               lhs_reordered_tokens, lhs_non_contracting_tokens,
                               contracting_tokens, batching_tokens, isLhsBNC);
    Value rhs = processOperand(rewriter, old_rhs, loc, rhs_original_tokens,
                               rhs_reordered_tokens, rhs_non_contracting_tokens,
                               contracting_tokens, batching_tokens, isRhsBNC);

    SmallVector<int64_t> lhs_contracting_dims;
    SmallVector<int64_t> rhs_contracting_dims;
    SmallVector<int64_t> batching_dims;
    auto lhs_type = lhs.getType().cast<RankedTensorType>();
    int64_t dot_rank = lhs_type.getRank();
    for (int64_t i = 0; i < batching_tokens.size(); ++i) {
      batching_dims.push_back(i);
    }
    int64_t lhs_contracting_dim = isLhsBNC ? dot_rank - 1 : dot_rank - 2;
    int64_t rhs_contracting_dim = isRhsBNC ? dot_rank - 1 : dot_rank - 2;
    lhs_contracting_dims.push_back(lhs_contracting_dim);
    rhs_contracting_dims.push_back(rhs_contracting_dim);
    auto dim_numbers = mhlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(), batching_dims, batching_dims,
        lhs_contracting_dims, rhs_contracting_dims);
    SmallVector<int64_t> dot_shape;
    auto lhs_shape = lhs_type.getShape();
    auto rhs_shape = rhs.getType().cast<RankedTensorType>().getShape();
    for (int64_t d = 0; d < dot_rank - 2; ++d) {
      dot_shape.push_back(lhs_shape[d]);
    }
    if (isLhsBNC) {
      dot_shape.push_back(lhs_shape[dot_rank - 2]);
    } else {
      dot_shape.push_back(lhs_shape[dot_rank - 1]);
    }
    if (isRhsBNC) {
      dot_shape.push_back(rhs_shape[dot_rank - 2]);
    } else {
      dot_shape.push_back(rhs_shape[dot_rank - 1]);
    }
    auto dot_type = RankedTensorType::get(dot_shape, lhs_type.getElementType());
    Value dot_result = rewriter.create<mhlo::DotGeneralOp>(
        loc, dot_type, lhs, rhs, dim_numbers, /*precision_config=*/ArrayAttr{});

    Value result = processResult(
        rewriter, loc, dot_result, old_lhs, old_rhs, contracting_tokens,
        batching_tokens, lhs_non_contracting_tokens, rhs_non_contracting_tokens,
        lhs_original_tokens, rhs_original_tokens, result_original_tokens,
        result_reordered_tokens);
    result.setType(op.getResult().getType());

    rewriter.replaceOp(op, result);
    return success();
  }

 private:
  // Analysis tokens, categorize into batching/contracting/non_contracting
  // tokens
  void categorizeTokens(
      const TokensTy& tokens, llvm::SmallDenseSet<char>& contracting_tokens,
      llvm::SmallDenseSet<char>& batching_tokens,
      llvm::SmallDenseSet<char>& lhs_non_contracting_tokens,
      llvm::SmallDenseSet<char>& rhs_non_contracting_tokens) const;
  // Reorder the tokens for lhs/rhs/result into a legalized layout
  void getReorderedTokens(
      const llvm::SmallDenseSet<char>& contracting_tokens,
      const llvm::SmallDenseSet<char>& batching_tokens,
      const llvm::SmallDenseSet<char>& lhs_non_contracting_tokens,
      const llvm::SmallDenseSet<char>& rhs_non_contracting_tokens,
      const SmallVector<char>& lhs_original_tokens,
      const SmallVector<char>& rhs_original_tokens,
      const SmallVector<char>& result_original_tokens, bool& isLhsBNC,
      bool& isRhsBNC, SmallVector<char>& lhs_reordered_tokens,
      SmallVector<char>& rhs_reordered_tokens,
      SmallVector<char>& result_reordered_tokens) const;
  // Insert potentialy needed transpose and reshape for lhs/rhs
  Value processOperand(PatternRewriter& rewriter, Value original_operand,
                       Location loc, const SmallVector<char>& original_tokens,
                       const SmallVector<char>& reordered_tokens,
                       const llvm::SmallDenseSet<char>& non_contracting_tokens,
                       const llvm::SmallDenseSet<char>& contracting_tokens,
                       const llvm::SmallDenseSet<char>& batching_tokens,
                       bool isBNC) const;
  Value processResult(
      PatternRewriter& rewriter, Location loc, Value dot_result, Value orig_lhs,
      Value orig_rhs, const llvm::SmallDenseSet<char>& contracting_tokens,
      const llvm::SmallDenseSet<char>& batching_tokens,
      const llvm::SmallDenseSet<char>& lhs_non_contracting_tokens,
      const llvm::SmallDenseSet<char>& rhs_non_contracting_tokens,
      const SmallVector<char>& lhs_original_tokens,
      const SmallVector<char>& rhs_original_tokens,
      const SmallVector<char>& result_original_tokens,
      const SmallVector<char>& result_reordered_tokens) const;
};

void EinsumToDotGeneralPattern::categorizeTokens(
    const TokensTy& tokens, llvm::SmallDenseSet<char>& contracting_tokens,
    llvm::SmallDenseSet<char>& batching_tokens,
    llvm::SmallDenseSet<char>& lhs_non_contracting_tokens,
    llvm::SmallDenseSet<char>& rhs_non_contracting_tokens) const {
  for (auto token : tokens) {
    // is a contracing dim token, if both lhs/rhs have it, but result doesn't
    // have it
    if (token.second.count(kIsLhs) > 0 && token.second.count(kIsRhs) > 0 &&
        token.second.count(kIsResult) == 0) {
      contracting_tokens.insert(token.first);
    } else if (token.second.count(kIsLhs) > 0 &&
               token.second.count(kIsRhs) > 0 &&
               token.second.count(kIsResult) > 0) {
      batching_tokens.insert(token.first);
    } else if (token.second.count(kIsLhs) > 0 &&
               token.second.count(kIsRhs) == 0 &&
               token.second.count(kIsResult) > 0) {
      lhs_non_contracting_tokens.insert(token.first);
    } else if (token.second.count(kIsLhs) == 0 &&
               token.second.count(kIsRhs) > 0 &&
               token.second.count(kIsResult) > 0) {
      rhs_non_contracting_tokens.insert(token.first);
    }
  }
}

// 1, batching dims, contracting dims, taking the order of lhs
// 2, non_contracting dims, taking the order associately from lhs/rhs
// 3, for lhs/rhs, if the last dim is contracting dim, the order will be
//    BNC: {batching_dims, non_contracting_dims, contracting_dims},
//    or else, the order will be:
//    BCN: {batching_dims, contracting_dims, non_contracting_dims}
void EinsumToDotGeneralPattern::getReorderedTokens(
    const llvm::SmallDenseSet<char>& contracting_tokens,
    const llvm::SmallDenseSet<char>& batching_tokens,
    const llvm::SmallDenseSet<char>& lhs_non_contracting_tokens,
    const llvm::SmallDenseSet<char>& rhs_non_contracting_tokens,
    const SmallVector<char>& lhs_original_tokens,
    const SmallVector<char>& rhs_original_tokens,
    const SmallVector<char>& result_original_tokens, bool& isLhsBNC,
    bool& isRhsBNC, SmallVector<char>& lhs_reordered_tokens,
    SmallVector<char>& rhs_reordered_tokens,
    SmallVector<char>& result_reordered_tokens) const {
  isLhsBNC = contracting_tokens.contains(lhs_original_tokens.back());
  isRhsBNC = contracting_tokens.contains(rhs_original_tokens.back());
  SmallVector<char> reordered_batching_tokens;
  SmallVector<char> reordered_contracting_tokens;
  SmallVector<char> reordered_lhs_non_contracting_tokens;
  SmallVector<char> reordered_rhs_non_contracting_tokens;
  for (char t : lhs_original_tokens) {
    if (batching_tokens.contains(t)) {
      reordered_batching_tokens.push_back(t);
    }
  }
  for (char t : lhs_original_tokens) {
    if (contracting_tokens.contains(t)) {
      reordered_contracting_tokens.push_back(t);
    }
  }
  for (char t : lhs_original_tokens) {
    if (lhs_non_contracting_tokens.contains(t)) {
      reordered_lhs_non_contracting_tokens.push_back(t);
    }
  }
  for (char t : rhs_original_tokens) {
    if (rhs_non_contracting_tokens.contains(t)) {
      reordered_rhs_non_contracting_tokens.push_back(t);
    }
  }

  // lhs/rhs
  std::copy(reordered_batching_tokens.begin(), reordered_batching_tokens.end(),
            std::back_inserter(lhs_reordered_tokens));
  std::copy(reordered_batching_tokens.begin(), reordered_batching_tokens.end(),
            std::back_inserter(rhs_reordered_tokens));
  if (isLhsBNC) {
    std::copy(reordered_lhs_non_contracting_tokens.begin(),
              reordered_lhs_non_contracting_tokens.end(),
              std::back_inserter(lhs_reordered_tokens));
    std::copy(reordered_contracting_tokens.begin(),
              reordered_contracting_tokens.end(),
              std::back_inserter(lhs_reordered_tokens));
  } else {
    std::copy(reordered_contracting_tokens.begin(),
              reordered_contracting_tokens.end(),
              std::back_inserter(lhs_reordered_tokens));
    std::copy(reordered_lhs_non_contracting_tokens.begin(),
              reordered_lhs_non_contracting_tokens.end(),
              std::back_inserter(lhs_reordered_tokens));
  }
  if (isRhsBNC) {
    std::copy(reordered_rhs_non_contracting_tokens.begin(),
              reordered_rhs_non_contracting_tokens.end(),
              std::back_inserter(rhs_reordered_tokens));
    std::copy(reordered_contracting_tokens.begin(),
              reordered_contracting_tokens.end(),
              std::back_inserter(rhs_reordered_tokens));
  } else {
    std::copy(reordered_contracting_tokens.begin(),
              reordered_contracting_tokens.end(),
              std::back_inserter(rhs_reordered_tokens));
    std::copy(reordered_rhs_non_contracting_tokens.begin(),
              reordered_rhs_non_contracting_tokens.end(),
              std::back_inserter(rhs_reordered_tokens));
  }
  // result
  std::copy(reordered_batching_tokens.begin(), reordered_batching_tokens.end(),
            std::back_inserter(result_reordered_tokens));
  std::copy(reordered_lhs_non_contracting_tokens.begin(),
            reordered_lhs_non_contracting_tokens.end(),
            std::back_inserter(result_reordered_tokens));
  std::copy(reordered_rhs_non_contracting_tokens.begin(),
            reordered_rhs_non_contracting_tokens.end(),
            std::back_inserter(result_reordered_tokens));
}

Value EinsumToDotGeneralPattern::processOperand(
    PatternRewriter& rewriter, Value original_operand, Location loc,
    const SmallVector<char>& original_tokens,
    const SmallVector<char>& reordered_tokens,
    const llvm::SmallDenseSet<char>& non_contracting_tokens,
    const llvm::SmallDenseSet<char>& contracting_tokens,
    const llvm::SmallDenseSet<char>& batching_tokens, bool isBNC) const {
  int64_t rank = reordered_tokens.size();
  auto orig_type = original_operand.getType().cast<RankedTensorType>();
  auto orig_shape = orig_type.getShape();
  Value result = original_operand;
  // If need transpose
  if (original_tokens != reordered_tokens) {
    SmallVector<int64_t> permutation;
    for (char t : reordered_tokens) {
      auto pos = std::find(original_tokens.begin(), original_tokens.end(), t);
      permutation.push_back(std::distance(original_tokens.begin(), pos));
    }
    auto permutation_attr = DenseIntElementsAttr::get(
        RankedTensorType::get({rank}, rewriter.getI64Type()), permutation);
    SmallVector<int64_t> transposed_shape;
    for (int64_t i : permutation) {
      transposed_shape.push_back(orig_shape[i]);
    }
    auto transposed_type =
        RankedTensorType::get(transposed_shape, orig_type.getElementType());
    result = rewriter.create<mhlo::TransposeOp>(
        loc, transposed_type, original_operand, permutation_attr);
  }
  size_t num_contracting_dims = contracting_tokens.size();
  size_t num_non_contracting_dims = non_contracting_tokens.size();
  // If a reshape is needed, aka, if num of contracting/non_contracting dims > 1
  if (num_contracting_dims > 1 || num_non_contracting_dims > 1) {
    SmallVector<SmallVector<int64_t>> reshape_maps;
    for (int64_t i = 0; i < batching_tokens.size(); ++i) {
      SmallVector<int64_t> b({i});
      reshape_maps.push_back(b);
    }
    SmallVector<int64_t> n;
    for (int64_t j = 0; j < non_contracting_tokens.size(); ++j) {
      n.push_back(batching_tokens.size() + j);
    }
    SmallVector<int64_t> c;
    for (int64_t j = 0; j < contracting_tokens.size(); ++j) {
      c.push_back(batching_tokens.size() + non_contracting_tokens.size() + j);
    }
    if (isBNC) {
      reshape_maps.push_back(n);
      reshape_maps.push_back(c);
    } else {
      reshape_maps.push_back(c);
      reshape_maps.push_back(n);
    }

    SmallVector<int64_t> reshaped_dims;
    SmallVector<Value> reshaped_shape_values;
    auto transposed_shape =
        result.getType().cast<RankedTensorType>().getShape();
    for (auto& dims : reshape_maps) {
      int64_t size_static = 1;
      Value size_value = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      for (auto dim : dims) {
        if (size_static == ShapedType::kDynamicSize ||
            transposed_shape[dim] == ShapedType::kDynamicSize) {
          size_static = ShapedType::kDynamicSize;
        } else {
          size_static *= transposed_shape[dim];
        }
        Value orig_dim_val =
            transposed_shape[dim] == ShapedType::kDynamicSize
                ? rewriter.create<tensor::DimOp>(loc, result, dim).getResult()
                : rewriter
                      .create<arith::ConstantIndexOp>(loc,
                                                      transposed_shape[dim])
                      .getResult();
        size_value =
            rewriter.create<arith::MulIOp>(loc, size_value, orig_dim_val);
      }
      reshaped_dims.push_back(size_static);
      reshaped_shape_values.push_back(size_value);
    }
    RankedTensorType reshaped_ty =
        RankedTensorType::get(reshaped_dims, orig_type.getElementType());
    Value reshaped_shape =
        rewriter.create<tensor::FromElementsOp>(loc, reshaped_shape_values);
    result = rewriter.create<mhlo::DynamicReshapeOp>(loc, reshaped_ty, result,
                                                     reshaped_shape);
  }
  return result;
}

Value EinsumToDotGeneralPattern::processResult(
    PatternRewriter& rewriter, Location loc, Value dot_result, Value orig_lhs,
    Value orig_rhs, const llvm::SmallDenseSet<char>& contracting_tokens,
    const llvm::SmallDenseSet<char>& batching_tokens,
    const llvm::SmallDenseSet<char>& lhs_non_contracting_tokens,
    const llvm::SmallDenseSet<char>& rhs_non_contracting_tokens,
    const SmallVector<char>& lhs_original_tokens,
    const SmallVector<char>& rhs_original_tokens,
    const SmallVector<char>& result_original_tokens,
    const SmallVector<char>& result_reordered_tokens) const {
  auto orig_lhs_shape = orig_lhs.getType().cast<RankedTensorType>().getShape();
  auto orig_rhs_shape = orig_rhs.getType().cast<RankedTensorType>().getShape();
  size_t num_contracting_dims = contracting_tokens.size();
  size_t num_lhs_non_contracting_dims = lhs_non_contracting_tokens.size();
  size_t num_rhs_non_contracting_dims = rhs_non_contracting_tokens.size();
  Value result = dot_result;
  auto dot_result_ty = dot_result.getType().cast<RankedTensorType>();
  // If a reshape is needed
  if (num_contracting_dims > 1 || num_lhs_non_contracting_dims > 1 ||
      num_rhs_non_contracting_dims > 1) {
    SmallVector<int64_t> reshaped_dims;
    SmallVector<Value> reshaped_shape_values;
    auto find_index = [&](const SmallVector<char>& vec, char token) -> int64_t {
      auto pos = std::find(vec.begin(), vec.end(), token);
      int64_t idx = std::distance(vec.begin(), pos);
      return idx;
    };
    for (char t : result_reordered_tokens) {
      // For a batching or a contracting dim, reshaped_dims /
      // reshaped_shape_values is taken from lhs. For non-contracting dims it
      // will be taken from lhs/rhs associately
      if (batching_tokens.contains(t) || contracting_tokens.contains(t) ||
          lhs_non_contracting_tokens.contains(t)) {
        int64_t lhs_idx = find_index(lhs_original_tokens, t);
        reshaped_dims.push_back(orig_lhs_shape[lhs_idx]);
        Value orig_dim_val =
            orig_lhs_shape[lhs_idx] == ShapedType::kDynamicSize
                ? rewriter.create<tensor::DimOp>(loc, orig_lhs, lhs_idx)
                      .getResult()
                : rewriter
                      .create<arith::ConstantIndexOp>(loc,
                                                      orig_lhs_shape[lhs_idx])
                      .getResult();
        reshaped_shape_values.push_back(orig_dim_val);
      } else if (rhs_non_contracting_tokens.contains(t)) {
        int64_t rhs_idx = find_index(rhs_original_tokens, t);
        reshaped_dims.push_back(orig_rhs_shape[rhs_idx]);
        Value orig_dim_val =
            orig_rhs_shape[rhs_idx] == ShapedType::kDynamicSize
                ? rewriter.create<tensor::DimOp>(loc, orig_rhs, rhs_idx)
                      .getResult()
                : rewriter
                      .create<arith::ConstantIndexOp>(loc,
                                                      orig_rhs_shape[rhs_idx])
                      .getResult();
        reshaped_shape_values.push_back(orig_dim_val);
      }
    }
    RankedTensorType reshaped_ty =
        RankedTensorType::get(reshaped_dims, dot_result_ty.getElementType());
    Value reshaped_shape =
        rewriter.create<tensor::FromElementsOp>(loc, reshaped_shape_values);
    result = rewriter.create<mhlo::DynamicReshapeOp>(loc, reshaped_ty, result,
                                                     reshaped_shape);
  }
  // If a transpose is needed
  if (result_original_tokens != result_reordered_tokens) {
    SmallVector<int64_t> permutation;
    for (char t : result_original_tokens) {
      auto pos = std::find(result_reordered_tokens.begin(),
                           result_reordered_tokens.end(), t);
      permutation.push_back(
          std::distance(result_reordered_tokens.begin(), pos));
    }
    auto permutation_attr = DenseIntElementsAttr::get(
        RankedTensorType::get({permutation.size()}, rewriter.getI64Type()),
        permutation);
    auto reshaped_shape = result.getType().cast<RankedTensorType>().getShape();
    SmallVector<int64_t> transposed_shape;
    for (int64_t i : permutation) {
      transposed_shape.push_back(reshaped_shape[i]);
    }
    auto transposed_type =
        RankedTensorType::get(transposed_shape, dot_result_ty.getElementType());
    result = rewriter.create<mhlo::TransposeOp>(loc, transposed_type, result,
                                                permutation_attr);
  }
  return result;
}

struct DotRewriterPass : public DotRewriterPassBase<DotRewriterPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    // TODO: if needs to do const reformat, we need the xla_hlo.dot with its
    // inputs

    MLIRContext* ctx = func.getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<DotToDotGeneralConvert, TransposeFoldingConvert,
                    EinsumToDotGeneralPattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      func.emitError("applyPatternsAndFoldGreedily does not converge");
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createDiscDotRewriterPass() {
  return std::make_unique<DotRewriterPass>();
}

}  // namespace disc_ral
}  // namespace mlir

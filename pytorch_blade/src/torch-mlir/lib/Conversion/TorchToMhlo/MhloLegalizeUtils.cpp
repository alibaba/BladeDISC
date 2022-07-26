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

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include <mlir-hlo/Dialect/mhlo/IR/hlo_ops.h> // from tf repo

#include <numeric>
#include "torch-mlir/Conversion/TorchToMhlo/MhloLegalizeUtils.h"

namespace mlir {
namespace mhlo {

std::vector<int64_t> rangeIndices(int64_t min, int64_t max) {
  if (min > max)
    std::swap(min, max);
  auto len = max - min;
  std::vector<int64_t> range(len);
  std::iota(range.begin(), range.end(), min);
  return range;
}

std::vector<int64_t> normalizeDimIndex(ArrayRef<int64_t> dims, int64_t rank) {
  std::vector<int64_t> newDims;
  newDims.reserve(rank);
  std::transform(
      dims.begin(),
      dims.end(),
      std::back_inserter(newDims),
      [rank](int64_t d) -> int64_t { return (d + rank) % rank; });
  return newDims;
}

// Create a 32-bit float constant operator from a float
Value getMhloConstTensorSingleF32(
    PatternRewriter& rewriter,
    Operation* op,
    float val) {
  auto constType = RankedTensorType::get({}, rewriter.getF32Type());
  auto constAttr = DenseElementsAttr::get(constType, val);

  auto const_op =
      rewriter.create<mhlo::ConstOp>(op->getLoc(), constType, constAttr);
  return const_op.getResult();
}

// Templated function to create a constant op for given type and shape.
// T: storage C type.
// Default template creates a constant tensor in T.
template <typename T>
llvm::Optional<Value> getConstTensor(
    PatternRewriter& rewriter,
    Operation* op,
    ArrayRef<T> vec,
    ArrayRef<int64_t> shape) {
  uint64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return llvm::None;
  }

  auto constType =
      RankedTensorType::get(shape, rewriter.getIntegerType(sizeof(T) * 8));
  auto constAttr = DenseElementsAttr::get(constType, vec);

  auto const_op =
      rewriter.create<mhlo::ConstOp>(op->getLoc(), constType, constAttr);
  return const_op.getResult();
}

// Template specialization for APInt
template <>
llvm::Optional<Value> getConstTensor<APInt>(
    PatternRewriter& rewriter,
    Operation* op,
    ArrayRef<APInt> vec,
    ArrayRef<int64_t> shape) {
  uint64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return llvm::None;
  }

  auto constType = RankedTensorType::get(
      shape, rewriter.getIntegerType(vec[0].getBitWidth()));
  auto constAttr = DenseElementsAttr::get(constType, vec);

  auto const_op =
      rewriter.create<mhlo::ConstOp>(op->getLoc(), constType, constAttr);
  return const_op.getResult();
}

// Template specialization for float
template <>
llvm::Optional<Value> getConstTensor<float>(
    PatternRewriter& rewriter,
    Operation* op,
    ArrayRef<float> vec,
    ArrayRef<int64_t> shape) {
  uint64_t num_total_elements = 1;
  for (int64_t a : shape) {
    num_total_elements *= a;
  }

  if (vec.size() != num_total_elements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return llvm::None;
  }

  auto constType = RankedTensorType::get(shape, rewriter.getF64Type());
  auto constAttr = DenseElementsAttr::get(constType, vec);

  auto const_op =
      rewriter.create<mhlo::ConstOp>(op->getLoc(), constType, constAttr);
  return const_op.getResult();
}

// Template instantiation
template llvm::Optional<Value> getConstTensor<int32_t>(
    PatternRewriter&,
    Operation*,
    ArrayRef<int32_t> vec,
    ArrayRef<int64_t> shape);

template llvm::Optional<Value> getConstTensor<int64_t>(
    PatternRewriter&,
    Operation*,
    ArrayRef<int64_t> vec,
    ArrayRef<int64_t> shape);

std::vector<Value> getDimSizesOfTensor(
    PatternRewriter& rewriter,
    Operation* op,
    Value value,
    ArrayRef<int64_t> inpDims) {
  auto currentKnowledge = ValueKnowledge::getKnowledgeFromType(value.getType());
  std::vector<Value> dimSizes;
  if (!currentKnowledge.hasRank) {
    op->emitOpError("getDimSizesOfTensor(): the input is not a ranked tensor");
    return dimSizes;
  }
  auto rank = currentKnowledge.sizes.size();
  if (rank == 0) {
    return dimSizes;
  }

  auto dims = normalizeDimIndex(inpDims, rank);
  dimSizes.reserve(rank);
  auto loc = op->getLoc();
  for (auto d : dims) {
    dimSizes.emplace_back(rewriter.create<arith::IndexCastOp>(
        loc,
        rewriter.getI32Type(),
        rewriter.create<tensor::DimOp>(loc, value, d)));
  }
  return dimSizes;
}

std::vector<Value> getDimSizesOfTensor(
    PatternRewriter& rewriter,
    Operation* op,
    Value value) {
  auto currentKnowledge = ValueKnowledge::getKnowledgeFromType(value.getType());
  std::vector<Value> dimSizes;
  if (!currentKnowledge.hasRank) {
    op->emitOpError("getDimSizesOfTensor(): the input is not a ranked tensor");
    return dimSizes;
  }
  auto rank = currentKnowledge.sizes.size();
  if (rank == 0) {
    return dimSizes;
  }

  dimSizes.reserve(rank);
  auto loc = op->getLoc();
  for (auto d = 0; d < rank; ++d) {
    dimSizes.emplace_back(rewriter.create<arith::IndexCastOp>(
        loc,
        rewriter.getI32Type(),
        rewriter.create<tensor::DimOp>(loc, value, d)));
  }
  return dimSizes;
}

llvm::Optional<Value> getMhloShapeOfTensor(
    PatternRewriter& rewriter,
    Operation* op,
    Value value) {
  auto dimSizes = getDimSizesOfTensor(rewriter, op, value);
  if (dimSizes.size() == 0) {
    return llvm::None;
  }
  return rewriter.create<tensor::FromElementsOp>(op->getLoc(), dimSizes)
      .getResult();
}

llvm::Optional<Value> getUnsqueezedTensor(
    PatternRewriter& rewriter,
    Operation* op,
    Value tensor,
    ArrayRef<int64_t> inputUnsqzDims) {
  // Returns a new tensor with dims of size 1 inserted at the specified
  // position.
  //
  // The position indices (must be high to low dimension number of the returned
  // tensor) are specified with unsqzDims. Indices must be in-order, and in
  // range of tensor rank. Thus, unsqueeze a rank 1 tensor with {0, 2}, {0, 1,
  // 3}, {0, 1, 2} are all valid dimension sets, but {0, 3}, {2} are not.
  auto dimSizes = getDimSizesOfTensor(rewriter, op, tensor);
  auto rank = dimSizes.size();
  size_t newRank = rank + inputUnsqzDims.size();
  auto unsqzDims = normalizeDimIndex(inputUnsqzDims, newRank);
  for (size_t k = 0; k < unsqzDims.size(); ++k) {
    if (k > 1 && unsqzDims[k] <= unsqzDims[k - 1]) {
      op->emitOpError("Unsqueeze dimensions must be specified in order.");
      return llvm::None;
    }
  }

  auto loc = op->getLoc();
  auto rankTy = tensor.getType().dyn_cast<RankedTensorType>();
  auto oldShape = rankTy.getShape();
  auto one =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(1));

  std::vector<Value> newDimSizes;
  std::vector<int64_t> newShape;
  newDimSizes.reserve(newRank);
  newShape.reserve(newRank);
  for (size_t k = 0, i = 0, j = 0; k < newRank; ++k) {
    if (j < unsqzDims.size() && unsqzDims[j] == k) {
      newDimSizes.push_back(one);
      newShape.push_back(1);
      j++;
    } else {
      newDimSizes.push_back(dimSizes[i]);
      newShape.push_back(oldShape[i]);
      i++;
    }
  }

  auto outTy = RankedTensorType::get(newShape, rankTy.getElementType());
  auto mhloShape = rewriter.create<tensor::FromElementsOp>(loc, newDimSizes);
  return rewriter.create<mhlo::DynamicReshapeOp>(loc, outTy, tensor, mhloShape)
      .getResult();
}

llvm::Optional<Value> getZeroRankTensor(
    PatternRewriter& rewriter,
    Operation* op,
    Value tensor) {
  auto rankTy = tensor.getType().dyn_cast<RankedTensorType>();
  if (!rankTy) {
    op->emitOpError("Could not reshape non ranked tensor to 0-rank");
    return llvm::None;
  }
  auto shape = rankTy.getShape();
  if (!(shape.size() == 1 && shape[0] == 1)) {
    op->emitOpError("Could not reshape non-rank 1 tensor to 0-rank");
    return llvm::None;
  }
  return rewriter
      .create<mhlo::ReshapeOp>(
          op->getLoc(),
          RankedTensorType::get(ArrayRef<int64_t>{}, rankTy.getElementType()),
          tensor)
      .getResult();
}

Value getNumelOfTensor(PatternRewriter& rewriter, Operation* op, Value value) {
  auto loc = op->getLoc();
  auto dimSizes = getDimSizesOfTensor(rewriter, op, value);
  Value numel =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(1));
  for (auto d : dimSizes) {
    numel = rewriter.create<arith::MulIOp>(loc, numel, d);
  }
  return numel;
}

Value getReshapedTensor(
    PatternRewriter& rewriter,
    Operation* op,
    Value tensor,
    ArrayRef<int64_t> shape,
    ArrayRef<Value> dimSizes) {
  // create mhlo::DynamicReshapeOp
  auto loc = op->getLoc();
  int newRank = dimSizes.size();
  auto tensorTy = tensor.getType().dyn_cast<RankedTensorType>();
  auto outRankTy = RankedTensorType::get(shape, tensorTy.getElementType());
  Value mhloShape = rewriter.create<tensor::FromElementsOp>(loc, dimSizes);
  return rewriter.create<mhlo::DynamicReshapeOp>(
      loc, outRankTy, tensor, mhloShape);
}

Value getExpandedTensor(
    PatternRewriter& rewriter,
    Operation* op,
    Value tensor,
    ArrayRef<Value> expandDimSizes,
    int64_t expandPos) {
  if (expandDimSizes.size() == 0) {
    return tensor;
  }

  auto tensorTy = tensor.getType().dyn_cast<RankedTensorType>();
  auto dimSizes = getDimSizesOfTensor(rewriter, op, tensor);
  int64_t rank = dimSizes.size();
  int64_t newRank = rank + expandDimSizes.size() - 1;
  expandPos = (expandPos + rank) % rank;

  std::vector<Value> newDimSizes;
  std::vector<int64_t> newShape;
  for (int64_t k = 0; k < rank; ++k) {
    if (k == expandPos) {
      newDimSizes.insert(
          newDimSizes.end(), expandDimSizes.begin(), expandDimSizes.end());
      for (size_t j = 0; j < expandDimSizes.size(); ++j) {
        newShape.push_back(ShapedType::kDynamicSize);
      }
    } else {
      newDimSizes.push_back(dimSizes[k]);
      newShape.push_back(tensorTy.getShape()[k]);
    }
  }

  return getReshapedTensor(rewriter, op, tensor, newShape, newDimSizes);
}

Value getProductOfDimSizes(
    PatternRewriter& rewriter,
    Operation* op,
    ArrayRef<Value> dimSizes) {
  auto loc = op->getLoc();
  auto prod =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(1))
          .getResult();

  for (auto& d : dimSizes) {
    prod = rewriter.create<arith::MulIOp>(loc, prod, d).getResult();
  }
  return prod;
}

std::tuple<Value, std::vector<Value>> getCollapsedTensor(
    PatternRewriter& rewriter,
    Operation* op,
    Value tensor,
    ArrayRef<int64_t> inpCollapDims) {
  // Ref to XLA:Collapse:
  // https://www.tensorflow.org/xla/operation_semantics#collapse However we use
  // high to low dimension indices.
  //
  // Collapse replaces the given subset of the operand's dimensions by a single
  // dimension. The input arguments are an arbitrary array of type T and a
  // compile-time-constant vector of dimension indices. The dimension indices
  // must be an in-order (high to low dimension numbers), consecutive subset of
  // T's dimensions. Thus, {0, 1, 2}, {0, 1}, or {1, 2} are all valid dimension
  // sets, but {1, 0} or {0, 2} are not.
  auto loc = op->getLoc();
  int64_t nCollaps = inpCollapDims.size();
  std::vector<Value> collapDimSizes;
  if (nCollaps == 0) {
    return std::make_tuple(tensor, collapDimSizes);
  }

  // CHECK the input collapse dimensions are in-order, otherwise throw exception
  auto tensorTy = tensor.getType().dyn_cast<RankedTensorType>();
  auto rank = tensorTy.getRank();
  std::vector<int64_t> collapDims = normalizeDimIndex(inpCollapDims, rank);
  for (size_t k = 1; k < nCollaps; ++k) {
    if (collapDims[k] != collapDims[k - 1] + 1)
      op->emitOpError("Collapse dims not in consecutive order");
  }

  // get original tensor shape in mlir standard dialect
  auto dimSizes = getDimSizesOfTensor(rewriter, op, tensor);

  // calculate the collapse new_dim, which build the graph in mlir standard
  // dialect
  for (auto k : collapDims) {
    auto dsize = dimSizes[k];
    collapDimSizes.push_back(dsize);
  }

  // gather the new dim size values
  std::vector<Value> newDimSizes;
  std::vector<int64_t> newShape;
  for (size_t k = 0; k < collapDims[0]; ++k) {
    newDimSizes.push_back(dimSizes[k]);
    newShape.push_back(tensorTy.getShape()[k]);
  }
  int64_t collapDimVal = 1;
  for (size_t k = collapDims[0]; k < collapDims[nCollaps - 1] + 1; ++k) {
    auto dsize = tensorTy.getShape()[k];
    if (dsize == ShapedType::kDynamicSize) {
      collapDimVal = ShapedType::kDynamicSize;
      break;
    }
    collapDimVal *= dsize;
  }
  newDimSizes.push_back(getProductOfDimSizes(rewriter, op, collapDimSizes));
  newShape.push_back(collapDimVal);
  for (size_t k = collapDims[nCollaps - 1] + 1; k < rank; ++k) {
    newDimSizes.push_back(dimSizes[k]);
    newShape.push_back(tensorTy.getShape()[k]);
  }

  return std::make_tuple(
      getReshapedTensor(rewriter, op, tensor, newShape, newDimSizes),
      collapDimSizes);
}

Value getBroadcastTensor(
    PatternRewriter& rewriter,
    Operation* op,
    Value tensor,
    ArrayRef<int64_t> shape,
    ArrayRef<Value> dimSizes,
    ArrayRef<int64_t> broadcastDims) {
  auto tensorTy = tensor.getType().dyn_cast<RankedTensorType>();
  int64_t rank = tensorTy.getRank();
  auto loc = op->getLoc();
  Value mhloShape = rewriter.create<tensor::FromElementsOp>(loc, dimSizes);

  RankedTensorType outTy =
      RankedTensorType::get(shape, tensorTy.getElementType());

  RankedTensorType attrTy = RankedTensorType::get(
      {static_cast<int64_t>(broadcastDims.size())},
      rewriter.getIntegerType(64));
  auto broadcastAttr = DenseIntElementsAttr::get(attrTy, broadcastDims);

  auto broadcast = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
      loc, outTy, tensor, mhloShape, broadcastAttr);
  return broadcast;
}

llvm::Optional<Value> getDotProduct(
    PatternRewriter& rewriter,
    Operation* op,
    Value lhs,
    Value rhs,
    int64_t rank) {
  if (rank < 2) {
    op->emitOpError("The input of DotProduct must has rank >= 2");
    return llvm::None;
  }

  std::vector<int64_t> batchDims;
  for (int64_t r = 0; r < rank - 2; ++r) {
    batchDims.push_back(r);
  }
  auto lhsTy = lhs.getType().dyn_cast<RankedTensorType>();
  auto rhsTy = rhs.getType().dyn_cast<RankedTensorType>();

  auto lhsShape = lhsTy.getShape();
  auto rhsShape = rhsTy.getShape();

  // lhsShape[b, m, n], rhsShape[b', n', k] -> resultShape[b, m, k],
  // assert b == b' and n == n', but we could only verify it at runtime
  std::vector<int64_t> resultShape(lhsShape.begin(), lhsShape.end());
  resultShape[rank - 1] = rhsShape[rank - 1];

  auto loc = op->getLoc();
  auto resultTy = RankedTensorType::get(resultShape, lhsTy.getElementType());
  auto dotDimAttr = mhlo::DotDimensionNumbersAttr::get(
      op->getContext(), batchDims, batchDims, {rank - 1}, {rank - 2});
  auto result = rewriter.create<mhlo::DotGeneralOp>(
      loc, resultTy, lhs, rhs, dotDimAttr, /*precision_config*/ nullptr);
  return result.getResult();
}

llvm::Optional<Value> getBmmDotProduct(
    PatternRewriter& rewriter,
    Operation* op,
    Value inpLhs,
    Value inpRhs) {
  Value lhs = inpLhs;
  Value rhs = inpRhs;
  auto lhsRankTy = inpLhs.getType().dyn_cast<RankedTensorType>();
  auto rhsRankTy = inpRhs.getType().dyn_cast<RankedTensorType>();

  auto lhsRank = lhsRankTy.getRank();
  auto rhsRank = rhsRankTy.getRank();
  if (lhsRank < 2) {
    op->emitOpError("The input of batch-matmul must has rank >= 2");
    return llvm::None;
  }
  if (rhsRank < 2) {
    op->emitOpError("The input of batch-matmul must has rank >= 2");
    return llvm::None;
  }

  // The non-matrix (i.e. batch) dimensions are broadcasted (and thus must be
  // broadcastable).
  auto maxRank = std::max(lhsRank, rhsRank);
  auto minRank = std::min(lhsRank, rhsRank);
  if (maxRank != minRank) {
    auto leadingRank = maxRank - minRank;
    auto leadingDims = rangeIndices(0, leadingRank);
    auto broadcastDims = rangeIndices(leadingRank, maxRank);
    auto lhsShape = lhsRankTy.getShape();
    auto rhsShape = rhsRankTy.getShape();
    if (lhsRank < rhsRank) {
      std::vector<int64_t> newShape(
          rhsShape.begin(), rhsShape.begin() + leadingRank);
      newShape.insert(newShape.end(), lhsShape.begin(), lhsShape.end());
      auto newDimSizes = getDimSizesOfTensor(rewriter, op, rhs, leadingDims);
      auto lhsDimSizes = getDimSizesOfTensor(rewriter, op, lhs);
      newDimSizes.insert(
          newDimSizes.end(), lhsDimSizes.begin(), lhsDimSizes.end());
      lhs = getBroadcastTensor(
          rewriter, op, lhs, newShape, newDimSizes, broadcastDims);
    } else {
      std::vector<int64_t> newShape(
          lhsShape.begin(), lhsShape.begin() + leadingRank);
      newShape.insert(newShape.end(), rhsShape.begin(), rhsShape.end());
      auto newDimSizes = getDimSizesOfTensor(rewriter, op, lhs, leadingDims);
      auto rhsDimSizes = getDimSizesOfTensor(rewriter, op, rhs);
      newDimSizes.insert(
          newDimSizes.end(), rhsDimSizes.begin(), rhsDimSizes.end());
      rhs = getBroadcastTensor(
          rewriter, op, rhs, newShape, newDimSizes, broadcastDims);
    }
  }

  // [?, ?, m, n] x [?, n, k] ==> batch_matmul([m,n], [n,k])
  return getDotProduct(rewriter, op, lhs, rhs, /*rank*/ maxRank);
}

llvm::Optional<Value> getMmDotProduct(
    PatternRewriter& rewriter,
    Operation* op,
    Value inpLhs,
    Value inpRhs) {
  auto lhsRankTy = inpLhs.getType().dyn_cast<RankedTensorType>();
  auto rhsRankTy = inpRhs.getType().dyn_cast<RankedTensorType>();

  auto lhsRank = lhsRankTy.getRank();
  auto rhsRank = rhsRankTy.getRank();
  if (lhsRank < 2) {
    op->emitOpError("The left hand-side input of matmul must has rank >= 2");
    return llvm::None;
  }
  if (rhsRank != 2) {
    op->emitOpError("The right hand-side input of matmul must has rank == 2");
    return llvm::None;
  }

  Value lhs = inpLhs;
  Value rhs = inpRhs;
  // [?, m, n] x [n, k] ==> [?xm, n] x [n, k]
  std::vector<Value> collapDimSizes;
  if (lhsRank > 2) {
    std::vector<int64_t> collapDims;
    for (size_t d = 0; d < lhsRank - 1; ++d) {
      collapDims.push_back(d);
    }
    std::tie(lhs, collapDimSizes) =
        getCollapsedTensor(rewriter, op, lhs, collapDims);
  }
  auto result = getDotProduct(rewriter, op, lhs, rhs, /*rank*/ 2);
  if (result) {
    return getExpandedTensor(
        rewriter, op, *result, collapDimSizes, /*expandPos*/ 0);
  }
  return llvm::None;
}

llvm::Optional<Value> getMvDotProduct(
    PatternRewriter& rewriter,
    Operation* op,
    Value inpLhs,
    Value inpRhs) {
  auto lhsRankTy = inpLhs.getType().dyn_cast<RankedTensorType>();
  auto rhsRankTy = inpRhs.getType().dyn_cast<RankedTensorType>();

  auto lhsRank = lhsRankTy.getRank();
  auto rhsRank = rhsRankTy.getRank();

  if (rhsRank != 1) {
    op->emitOpError("The right hand-side input of matmul must has rank == 1");
    return llvm::None;
  }
  if (lhsRank < 2) {
    op->emitOpError("The left hand-side input of matmul must has rank >= 2");
    return llvm::None;
  }

  auto unsqzRhs = getUnsqueezedTensor(rewriter, op, inpRhs, {1});
  if (!unsqzRhs) {
    return llvm::None;
  }
  auto product = getMmDotProduct(rewriter, op, inpLhs, *unsqzRhs);
  if (product) {
    Value result = *product;
    std::vector<Value> collapDimSizes;
    std::tie(result, collapDimSizes) =
        getCollapsedTensor(rewriter, op, result, {-2, -1});
    return result;
  }
  return llvm::None;
}

Value getPermutedTensor(
    PatternRewriter& rewriter,
    Operation* op,
    Value input,
    ArrayRef<int64_t> inpTransDims) {
  auto inputTy = input.getType().dyn_cast<RankedTensorType>();
  auto rank = inputTy.getRank();
  auto transDims = normalizeDimIndex(inpTransDims, rank);
  auto inpShape = inputTy.getShape();
  std::vector<int64_t> newShape;
  newShape.reserve(rank);

  for (auto d : transDims) {
    newShape.push_back(inpShape[d]);
  }

  auto attrTy = RankedTensorType::get(
      {static_cast<int64_t>(transDims.size())}, rewriter.getIntegerType(64));
  auto permuteAttr = DenseIntElementsAttr::get(attrTy, transDims);

  auto outTy = RankedTensorType::get(newShape, inputTy.getElementType());
  auto result = rewriter.create<mhlo::TransposeOp>(
      op->getLoc(), outTy, input, permuteAttr);
  return result.getResult();
}

Value getNormalizedDimSizeInternal(
    PatternRewriter& rewriter,
    Operation* op,
    Value index,
    Value dimSize) {
  auto loc = op->getLoc();
  Value zero = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIntegerAttr(rewriter.getI64Type(), 0));
  Value one = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIntegerAttr(rewriter.getI64Type(), 1));

  // To normalize index into range [-dimSize, dimSize]
  // index = min(max(-dimSize, index), dimSize)
  auto negDimSize = rewriter.create<arith::SubIOp>(loc, zero, dimSize);
  index = rewriter.create<arith::MaxSIOp>(loc, negDimSize, index);
  index = rewriter.create<arith::MinSIOp>(loc, dimSize, index);

  auto dimSizePlusIndex = rewriter.create<arith::AddIOp>(loc, dimSize, index);
  auto indexPositive = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::sge, index, zero);
  // get positive index: (index >=0) ? index: index + dimSize
  return rewriter.create<arith::SelectOp>(
      loc, indexPositive, index, dimSizePlusIndex);
}

Value getDynamicSliceInternal(
    PatternRewriter& rewriter,
    Operation* op,
    Value input,
    Value startIndex,
    Value endIndex,
    Value step,
    size_t dimIndex,
    ArrayRef<Value> dimSizes) {
  auto loc = op->getLoc();
  // startIndex & endIndex has been normailized into range [0, dSize]
  Type i32Type = rewriter.getI32Type();

  Value zero = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIntegerAttr(i32Type, 0));
  Value one = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIntegerAttr(i32Type, 1));

  SmallVector<Value, 4> startIndices;
  SmallVector<Value, 4> endIndices;
  SmallVector<Value, 4> strides;

  auto inputTy = input.getType().dyn_cast<RankedTensorType>();
  auto rank = inputTy.getRank();
  startIndices.reserve(rank);
  endIndices.reserve(rank);
  strides.reserve(rank);

  auto endIndexIsZero = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, endIndex, zero);
  endIndex = rewriter.create<arith::SelectOp>(
      loc, endIndexIsZero, dimSizes[dimIndex], endIndex);

  for (size_t r = 0; r < rank; ++r) {
    if (r == dimIndex) {
      startIndices.push_back(startIndex);
      endIndices.push_back(endIndex);
      strides.push_back(step);
    } else {
      startIndices.push_back(zero);
      endIndices.push_back(dimSizes[r]);
      strides.push_back(one);
    }
  }

  auto startTensor =
      rewriter.create<tensor::FromElementsOp>(loc, startIndices).getResult();
  auto endTensor =
      rewriter.create<tensor::FromElementsOp>(loc, endIndices).getResult();
  auto stridesTensor =
      rewriter.create<tensor::FromElementsOp>(loc, strides).getResult();

  auto inputShape = inputTy.getShape();
  SmallVector<int64_t, 4> sliceShape(inputShape.begin(), inputShape.end());
  sliceShape[dimIndex] = ShapedType::kDynamicSize;
  auto sliceoutputTy =
      RankedTensorType::get(sliceShape, inputTy.getElementType());
  return rewriter.create<mhlo::RealDynamicSliceOp>(
      loc, sliceoutputTy, input, startTensor, endTensor, stridesTensor);
}

Value getDynamicSlice(
    PatternRewriter& rewriter,
    Operation* op,
    Value input,
    llvm::Optional<Value> startIndexOpt,
    llvm::Optional<Value> endIndexOpt,
    llvm::Optional<Value> stepOpt,
    int64_t dim) {
  auto loc = op->getLoc();
  auto inputTy = input.getType().dyn_cast<RankedTensorType>();
  auto rank = inputTy.getRank();

  dim = (dim + rank) % rank;
  Value dimSize = rewriter.create<arith::IndexCastOp>(
      loc,
      rewriter.getI64Type(),
      rewriter.create<tensor::DimOp>(loc, input, dim));

  Value normStartIndex = startIndexOpt
      ? getNormalizedDimSizeInternal(rewriter, op, *startIndexOpt, dimSize)
      : rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIntegerAttr(rewriter.getI64Type(), 0));
  Value normEndIndex = endIndexOpt
      ? getNormalizedDimSizeInternal(rewriter, op, *endIndexOpt, dimSize)
      : dimSize;
  Value step = stepOpt
      ? *stepOpt
      : rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIntegerAttr(rewriter.getI64Type(), 1));

  auto i32Type = rewriter.getI32Type();
  auto startIndexI32 =
      rewriter.create<arith::TruncIOp>(loc, i32Type, normStartIndex);
  auto endIndexI32 =
      rewriter.create<arith::TruncIOp>(loc, i32Type, normEndIndex);
  auto stepI32 = rewriter.create<arith::TruncIOp>(loc, i32Type, step);
  auto dimSizes = getDimSizesOfTensor(rewriter, op, input);

  return getDynamicSliceInternal(
      rewriter, op, input, startIndexI32, endIndexI32, stepI32, dim, dimSizes);
}

Value getGatheredTensor(
    PatternRewriter& rewriter,
    Operation* op,
    Value params,
    Value indices,
    int64_t axis) {
  auto loc = op->getLoc();
  auto i32Type = rewriter.getI32Type();
  Value one = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIntegerAttr(i32Type, 1));

  // sliceSizes
  auto paramsRankTy = params.getType().dyn_cast<RankedTensorType>();
  auto paramsRank = paramsRankTy.getRank();
  SmallVector<Value, 4> sliceSizes;
  sliceSizes.reserve(paramsRank);
  for (int64_t r = 0; r < paramsRank; ++r) {
    if (r == axis) {
      sliceSizes.push_back(one);
    } else {
      sliceSizes.push_back(rewriter.create<arith::IndexCastOp>(
          loc, i32Type, rewriter.create<tensor::DimOp>(loc, params, r)));
    }
  }
  auto sliceSizesTensor =
      rewriter.create<tensor::FromElementsOp>(loc, sliceSizes);

  // offsetDims
  SmallVector<int64_t, 4> offsetDims;
  offsetDims.reserve(paramsRank);
  for (int64_t r = 0; r < axis; ++r) {
    offsetDims.push_back(r);
  }
  auto indicesRankTy = indices.getType().dyn_cast<RankedTensorType>();
  auto indicesRank = indicesRankTy.getRank();
  for (int64_t r = axis + 1; r < paramsRank; ++r) {
    offsetDims.push_back(r + indicesRank - 1);
  }

  // collapsedSliceDims
  SmallVector<int64_t, 4> collapsedSliceDims(1, axis);
  // startIndexMap
  SmallVector<int64_t, 4> startIndexMap(1, axis);
  // indexVecDim
  int64_t indexVecDim = indicesRank;
  auto dimsAttr = GatherDimensionNumbersAttr::get(
      rewriter.getContext(),
      /*offsetDims=*/offsetDims,
      /*collapsedSliceDims=*/collapsedSliceDims,
      /*startIndexMap=*/startIndexMap,
      /*indexVecDim=*/indexVecDim);

  // outputShape = params.shape[:axis] + indices.shape +
  //                params.shape[axis + 1:]
  auto paramsShape = paramsRankTy.getShape();
  auto indicesShape = indicesRankTy.getShape();
  SmallVector<int64_t, 4> outputShape(
      paramsShape.begin(), paramsShape.begin() + axis);
  outputShape.insert(
      outputShape.end(), indicesShape.begin(), indicesShape.end());
  outputShape.insert(
      outputShape.end(), paramsShape.begin() + axis + 1, paramsShape.end());

  // create output tensor type
  auto outputTy =
      RankedTensorType::get(outputShape, paramsRankTy.getElementType());
  return rewriter
      .create<mhlo::DynamicGatherOp>(
          loc, outputTy, params, indices, sliceSizesTensor, dimsAttr)
      .getResult();
}

mlir::Value getRollTensor(
    PatternRewriter& rewriter,
    Operation* op,
    mlir::Value input,
    int64_t shiftInt,
    int64_t dim) {
  // roll(input, shift, dim) = cat(
  //   slice(input, (dimSize - shift) % dimSize, dimSize),
  //   slice(input, 0, - shift))
  auto loc = op->getLoc();
  Type i32Type = rewriter.getI32Type();
  auto dimSizes = getDimSizesOfTensor(rewriter, op, input);
  auto dimSize = dimSizes[dim];

  Value zero =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
  Value one =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(1));
  Value shift = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32IntegerAttr(shiftInt));
  Value splitPos = rewriter.create<arith::SubIOp>(loc, dimSize, shift);
  splitPos = rewriter.create<arith::RemSIOp>(loc, splitPos, dimSize);

  auto lhs = getDynamicSliceInternal(
      rewriter, op, input, splitPos, dimSize, one, dim, dimSizes);
  auto rhs = getDynamicSliceInternal(
      rewriter, op, input, zero, splitPos, one, dim, dimSizes);
  auto result = rewriter.create<mhlo::ConcatenateOp>(
      loc,
      input.getType(),
      ArrayRef<Value>{lhs, rhs},
      rewriter.getI64IntegerAttr(dim));
  return result;
}

} // namespace mhlo
} // namespace mlir

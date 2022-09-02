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
#include <numeric>
#include "torch-mlir/Conversion/TorchToMhlo/MhloLegalizeUtils.h"
#include "torch-mlir/Conversion/TorchToMhlo/TorchToMhlo.h"

#include <mlir-hlo/Dialect/mhlo/IR/chlo_ops.h> // from tf repo
#include <mlir-hlo/Dialect/mhlo/IR/hlo_ops.h> // from tf repo

#include "../PassDetail.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

#include "torch-mlir/Conversion/MhloPasses.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

#include "tensorflow/compiler/mlir/disc/IR/hlo_disc_ops.h"
#include "tensorflow/compiler/mlir/disc/IR/rng_uniform_custom_call_op.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::TorchConversion;

static constexpr size_t kMhloDimSizeBits = 32;

namespace mlir {
namespace torch {
namespace TorchConversion {
static Value createInitialValueForReduceOp(
    Operation* op,
    Type elementTy,
    PatternRewriter& rewriter) {
  auto constType = RankedTensorType::get({}, elementTy);
  if (isa<AtenSumOp, AtenSumDimIntListOp>(op)) {
    if (elementTy.isa<mlir::FloatType>()) {
      auto constAttr = DenseElementsAttr::get(
          constType,
          {APFloat::getZero(
              elementTy.cast<mlir::FloatType>().getFloatSemantics(),
              /*negative=*/false)});
      return rewriter.create<mhlo::ConstantOp>(
          op->getLoc(), constType, constAttr);
    } else if (
        elementTy.isa<mlir::IntegerType>() &&
        elementTy.getIntOrFloatBitWidth() != 8) {
      auto constAttr = DenseElementsAttr::get(
          constType, {APInt::getZero(elementTy.getIntOrFloatBitWidth())});
      return rewriter.create<mhlo::ConstantOp>(
          op->getLoc(), constType, constAttr);
    }
  }

  if (isa<AtenMaxOp, AtenMaxDimOp, AtenArgmaxOp>(op)) {
    if (elementTy.isa<mlir::FloatType>()) {
      auto constAttr = DenseElementsAttr::get(
          constType,
          {APFloat::getLargest(
              elementTy.cast<mlir::FloatType>().getFloatSemantics(),
              /*negative=*/true)});
      return rewriter.create<mhlo::ConstantOp>(
          op->getLoc(), constType, constAttr);
    } else if (
        elementTy.isa<mlir::IntegerType>() &&
        elementTy.getIntOrFloatBitWidth() != 8) {
      auto constAttr = DenseElementsAttr::get(
          constType,
          {APInt::getSignedMinValue(elementTy.getIntOrFloatBitWidth())});
      return rewriter.create<mhlo::ConstantOp>(
          op->getLoc(), constType, constAttr);
    }
  }

  op->emitError(
      "unimplemented lowering in "
      "createInitialValueForReduceOp");
  return nullptr;
}

template <typename AtenOpT>
class ConvertAtenOp : public OpConversionPattern<AtenOpT> {
 public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult matchAndRewrite(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override;
};
int64_t toPositiveDim(int64_t dim, int64_t inputRank) {
  return dim >= 0 ? dim : dim + inputRank;
}

SmallVector<size_t> toPositiveDims(ArrayRef<int64_t> dims, int64_t rank) {
  SmallVector<size_t> posDims;
  posDims.reserve(rank);
  std::transform(
      dims.begin(),
      dims.end(),
      std::back_inserter(posDims),
      [rank](int64_t d) -> size_t { return toPositiveDim(d, rank); });
  return posDims;
}

FailureOr<SmallVector<Value, 4>> getDimSizesOfTensor2(
    PatternRewriter& rewriter,
    Operation* op,
    Value value,
    ArrayRef<int64_t> inpDims) {
  auto valueTy = value.getType().dyn_cast<RankedTensorType>();
  if (!valueTy) {
    return rewriter.notifyMatchFailure(
        op, "getDimSizesOfTensor(): the input is not a ranked tensor");
  }

  auto rank = valueTy.getRank();
  auto dims = toPositiveDims(inpDims, rank);
  SmallVector<Value, 4> dimSizes;
  dimSizes.reserve(dims.size());

  auto loc = op->getLoc();
  for (auto d : dims) {
    dimSizes.emplace_back(rewriter.create<arith::IndexCastOp>(
        loc,
        rewriter.getIntegerType(kMhloDimSizeBits),
        rewriter.create<tensor::DimOp>(loc, value, d)));
  }
  return dimSizes;
}

FailureOr<SmallVector<Value, 4>> getDimSizesOfTensor2(
    PatternRewriter& rewriter,
    Operation* op,
    Value value) {
  auto valueTy = value.getType().dyn_cast<RankedTensorType>();
  if (!valueTy) {
    return rewriter.notifyMatchFailure(
        op, "getDimSizesOfTensor(): the input is not a ranked tensor");
  }

  auto rank = valueTy.getRank();
  // Get int vector [0, 1, ..., rank-1]
  std::vector<int64_t> dims(rank);
  std::iota(dims.begin(), dims.end(), 0);
  return getDimSizesOfTensor2(rewriter, op, value, dims);
}

static llvm::Optional<ValueRange> getMaxValueInDim(
    ConversionPatternRewriter& rewriter,
    Operation* op,
    Value& input,
    int64_t dim) {
  auto inputTy = input.getType().template cast<RankedTensorType>();
  if (!inputTy) {
    return llvm::None;
  }
  if (!inputTy.getElementType().isIntOrFloat()) {
    return llvm::None;
  }
  auto inputElemTy = inputTy.getElementType();

  Value initValue = createInitialValueForReduceOp(op, inputElemTy, rewriter);
  if (!initValue)
    return llvm::None;

  DenseIntElementsAttr dimensions = DenseIntElementsAttr::get(
      RankedTensorType::get({}, rewriter.getI64Type()), dim);

  // value reduction
  auto valueReduceOp = rewriter.create<mhlo::ReduceOp>(
      op->getLoc(), input, initValue, dimensions);
  {
    Block& block = valueReduceOp.body().emplaceBlock();
    auto argumentType = RankedTensorType::get({}, inputTy.getElementType());
    block.addArgument(argumentType, op->getLoc());
    block.addArgument(argumentType, op->getLoc());
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&block);
      auto retValue =
          rewriter
              .create<mhlo::MaxOp>(
                  op->getLoc(), block.getArgument(0), block.getArgument(1))
              .getResult();
      rewriter.create<mhlo::ReturnOp>(op->getLoc(), retValue);
    }
  }
  return valueReduceOp.getResults();
}

static llvm::Optional<ValueRange> getMaxIndicesInDim(
    ConversionPatternRewriter& rewriter,
    Operation* op,
    Value& input,
    ArrayRef<Value> inputShapeVec,
    int64_t dim) {
  auto inputTy = input.getType().template cast<RankedTensorType>();
  if (!inputTy) {
    return llvm::None;
  }
  if (!inputTy.getElementType().isIntOrFloat()) {
    return llvm::None;
  }
  auto inputShape = inputTy.getShape();
  auto inputElemTy = inputTy.getElementType();

  Value initValue = createInitialValueForReduceOp(op, inputElemTy, rewriter);
  if (!initValue)
    return llvm::None;
  auto initIndex =
      mhlo::getConstTensor<int32_t>(rewriter, op, {0}, {}).getValue();

  DenseIntElementsAttr dimensions = DenseIntElementsAttr::get(
      RankedTensorType::get({}, rewriter.getI64Type()), dim);

  auto inputShapeTensor = rewriter.create<mlir::tensor::FromElementsOp>(
      op->getLoc(), inputShapeVec);

  auto indexTensor = rewriter.create<mhlo::DynamicIotaOp>(
      op->getLoc(),
      RankedTensorType::get(
          inputShape, rewriter.getIntegerType(kMhloDimSizeBits)),
      inputShapeTensor,
      rewriter.getI64IntegerAttr(dim));
  auto indicesReduceOp = rewriter.create<mhlo::ReduceOp>(
      op->getLoc(),
      ValueRange{input, indexTensor},
      ValueRange{
          initValue,
          initIndex,
      },
      dimensions);
  {
    Block& block = indicesReduceOp.body().emplaceBlock();

    // Add block arguments
    auto blockValArgumentType =
        RankedTensorType::get({}, inputTy.getElementType());
    auto blockIdxArgumentType =
        RankedTensorType::get({}, rewriter.getIntegerType(kMhloDimSizeBits));
    auto compareResultType = RankedTensorType::get({}, rewriter.getI1Type());
    block.addArgument(blockValArgumentType, op->getLoc());
    block.addArgument(blockIdxArgumentType, op->getLoc());

    block.addArgument(blockValArgumentType, op->getLoc());
    block.addArgument(blockIdxArgumentType, op->getLoc());

    auto* firstValArg = block.args_begin();
    auto* firstIdxArg = std::next(firstValArg);
    auto* secondValArg = std::next(firstIdxArg);
    auto* secondIdxArg = std::next(secondValArg);

    mhlo::ComparisonTypeAttr compareTypeAttr;
    if (inputTy.getElementType().isa<mlir::FloatType>()) {
      compareTypeAttr = mhlo::ComparisonTypeAttr::get(
          rewriter.getContext(), mhlo::ComparisonType::FLOAT);
    } else if (inputTy.getElementType().isa<mlir::IntegerType>()) {
      compareTypeAttr = mhlo::ComparisonTypeAttr::get(
          rewriter.getContext(), mhlo::ComparisonType::SIGNED);
    }
    mhlo::ComparisonDirectionAttr compareGeDirectionAttr =
        mhlo::ComparisonDirectionAttr::get(
            rewriter.getContext(), mhlo::ComparisonDirection::GE);
    mhlo::ComparisonDirectionAttr compareEqDirectionAttr =
        mhlo::ComparisonDirectionAttr::get(
            rewriter.getContext(), mhlo::ComparisonDirection::EQ);

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&block);

      Value compareGeResult = rewriter.create<mhlo::CompareOp>(
          op->getLoc(),
          compareResultType,
          *firstValArg,
          *secondValArg,
          compareGeDirectionAttr,
          compareTypeAttr);
      Value retValResult = rewriter.create<mhlo::SelectOp>(
          op->getLoc(), compareGeResult, *firstValArg, *secondValArg);

      // get smaller index value if compared nums are equal.
      Value compareEqResult = rewriter.create<mhlo::CompareOp>(
          op->getLoc(),
          compareResultType,
          *firstValArg,
          *secondValArg,
          compareEqDirectionAttr,
          compareTypeAttr);
      Value minIdx = rewriter.create<mhlo::MinOp>(
          op->getLoc(), *firstIdxArg, *secondIdxArg);
      Value idxWithGeVal = rewriter.create<mhlo::SelectOp>(
          op->getLoc(), compareGeResult, *firstIdxArg, *secondIdxArg);
      Value retIdxResult = rewriter.create<mhlo::SelectOp>(
          op->getLoc(), compareEqResult, minIdx, idxWithGeVal);

      rewriter.create<mhlo::ReturnOp>(
          op->getLoc(), mlir::ValueRange{retValResult, retIdxResult});
    }
  }
  return indicesReduceOp.getResults();
}

template <>
LogicalResult ConvertAtenOp<AtenMaxDimOp>::matchAndRewrite(
    AtenMaxDimOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  Value input = adaptor.self();
  auto inputTy = input.getType().template dyn_cast<RankedTensorType>();
  if (!inputTy) {
    return rewriter.notifyMatchFailure(
        op, "only Tensor types supported in MHLO");
  }
  auto inputElemTy = inputTy.getElementType();
  if (!inputElemTy.isIntOrFloat()) {
    return op.emitError(
        "Only floating-point or integer datatype legalization supported");
  }
  // Currently, (u)int8 dtype is not supported
  if (inputElemTy.isa<mlir::IntegerType>() &&
      inputElemTy.getIntOrFloatBitWidth() == 8) {
    return rewriter.notifyMatchFailure(
        op,
        "IntegerType with bitwidth 8 unsupported in convertion from "
        "AtenMaxDimOp to MHLO");
  }

  RankedTensorType valResultType = getTypeConverter()
                                       ->convertType(op.getResult(0).getType())
                                       .template cast<RankedTensorType>();
  RankedTensorType idxResultType = getTypeConverter()
                                       ->convertType(op.getResult(1).getType())
                                       .template cast<RankedTensorType>();
  Type idxElementType = idxResultType.getElementType();
  if (!idxElementType.isa<mlir::IntegerType>()) {
    return op.emitError("Aten.max.dim needs integer-like result");
  }

  int64_t dim;
  if (!matchPattern(op.dim(), m_TorchConstantInt(&dim))) {
    return rewriter.notifyMatchFailure(op, "non-int dim unsupported");
  }
  dim = toPositiveDim(dim, inputTy.getRank());
  if (!isValidDim(dim, inputTy.getRank())) {
    return rewriter.notifyMatchFailure(op, "dim is not a valid dim");
  }
  bool keepDim = false;
  if (!matchPattern(op.keepdim(), m_TorchConstantBool(&keepDim))) {
    return rewriter.notifyMatchFailure(op, "non-bool keepdim unsupported");
  }

  auto inputShapeInfo = getDimSizesOfTensor2(rewriter, op, input);
  if (failed(inputShapeInfo)) {
    return rewriter.notifyMatchFailure(
        op, "failed to get dimension sizes of the input");
  }
  auto inputShapeVec = *inputShapeInfo;
  auto valueResult = getMaxValueInDim(rewriter, op, input, dim).getValue()[0];
  auto indicesResult =
      getMaxIndicesInDim(rewriter, op, input, inputShapeVec, dim).getValue()[1];
  if (keepDim) {
    auto outShapeVec = inputShapeVec;
    outShapeVec[dim] = rewriter.create<mlir::arith::ConstantOp>(
        op->getLoc(),
        rewriter.getIntegerAttr(rewriter.getIntegerType(kMhloDimSizeBits), 1));
    auto outShapeTensor = rewriter.create<mlir::tensor::FromElementsOp>(
        op->getLoc(), outShapeVec);
    auto mhloReduceValueResult = rewriter.create<mhlo::DynamicReshapeOp>(
        op->getLoc(), valResultType, valueResult, outShapeTensor);
    auto mhloReduceIndexResult = rewriter.create<mhlo::DynamicReshapeOp>(
        op->getLoc(), idxResultType, indicesResult, outShapeTensor);
    rewriter.replaceOp(op, {mhloReduceValueResult, mhloReduceIndexResult});
    return success();
  }
  rewriter.replaceOp(op, {valueResult, indicesResult});
  return success();
}

Value gatherTensorAlongSingleAxis(
    PatternRewriter& rewriter,
    Operation* op,
    Value input,
    Value indices,
    int64_t axis) {
  auto loc = op->getLoc();
  Type intType = rewriter.getIntegerType(kMhloDimSizeBits);
  Value one = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIntegerAttr(intType, 1));

  // sliceSizes
  auto inputRankTy = input.getType().dyn_cast<RankedTensorType>();
  auto inputRank = inputRankTy.getRank();
  SmallVector<Value, 4> sliceSizes;
  sliceSizes.reserve(inputRank);
  for (int64_t r = 0; r < inputRank; ++r) {
    if (r == axis) {
      sliceSizes.push_back(one);
    } else {
      sliceSizes.push_back(rewriter.create<arith::IndexCastOp>(
          loc, intType, rewriter.create<tensor::DimOp>(loc, input, r)));
    }
  }
  auto sliceSizesTensor =
      rewriter.create<tensor::FromElementsOp>(loc, sliceSizes);

  // offsetDims
  SmallVector<int64_t, 4> offsetDims;
  offsetDims.reserve(inputRank);
  for (int64_t r = 0; r < axis; ++r) {
    offsetDims.push_back(r);
  }
  auto indicesRankTy = indices.getType().dyn_cast<RankedTensorType>();
  auto indicesRank = indicesRankTy.getRank();
  for (int64_t r = axis + 1; r < inputRank; ++r) {
    offsetDims.push_back(r + indicesRank - 1);
  }

  // collapsedSliceDims
  SmallVector<int64_t, 4> collapsedSliceDims(1, axis);
  // startIndexMap
  SmallVector<int64_t, 4> startIndexMap(1, axis);
  // indexVecDim
  int64_t indexVecDim = indicesRank;
  auto dimsAttr = mhlo::GatherDimensionNumbersAttr::get(
      rewriter.getContext(),
      /*offsetDims=*/offsetDims,
      /*collapsedSliceDims=*/collapsedSliceDims,
      /*startIndexMap=*/startIndexMap,
      /*indexVecDim=*/indexVecDim);

  // outputShape = input.shape[:axis] + indices.shape +
  //                input.shape[axis + 1:]
  auto inputShape = inputRankTy.getShape();
  auto indicesShape = indicesRankTy.getShape();
  SmallVector<int64_t, 4> outputShape(
      inputShape.begin(), inputShape.begin() + axis);
  outputShape.insert(
      outputShape.end(), indicesShape.begin(), indicesShape.end());
  outputShape.insert(
      outputShape.end(), inputShape.begin() + axis + 1, inputShape.end());

  // create output tensor type
  auto outputTy =
      RankedTensorType::get(outputShape, inputRankTy.getElementType());
  return rewriter
      .create<mhlo::DynamicGatherOp>(
          loc, outputTy, input, indices, sliceSizesTensor, dimsAttr)
      .getResult();
}

// Ref:
// https://pytorch.org/docs/stable/generated/torch.nn.functional.embedding.html
// padding_idx (int, optional)
//  – If specified, the entries at padding_idx do not contribute to the
//  gradient; therefore, the embedding vector at padding_idx is not updated
//  during training, i.e. it remains as a fixed “pad”.
// scale_grad_by_freq (boolean, optional)
//  – If given, this will scale gradients by the inverse of frequency of the
//  words in the mini-batch. Default False.
// sparse (bool, optional)
//  – If True, gradient w.r.t. weight matrix will be a sparse tensor.
template <>
LogicalResult ConvertAtenOp<AtenEmbeddingOp>::matchAndRewrite(
    AtenEmbeddingOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto weight = adaptor.weight();
  auto weightTy = weight.getType().template cast<RankedTensorType>();
  if (!weightTy)
    return op.emitError("only ranked tensor types are supported");

  int64_t padding_idx;
  if (!matchPattern(op.padding_idx(), m_TorchConstantInt(&padding_idx)))
    return rewriter.notifyMatchFailure(
        op, "only constant padding_idx is currently supported");

  bool scale_grad_by_freq;
  if (!matchPattern(
          op.scale_grad_by_freq(), m_TorchConstantBool(&scale_grad_by_freq)))
    return rewriter.notifyMatchFailure(
        op, "only constant scale_grad_by_freq is currently supported");
  if (scale_grad_by_freq)
    return rewriter.notifyMatchFailure(
        op, "scale gradients is currently not supported");
  bool sparse;
  if (!matchPattern(op.sparse(), m_TorchConstantBool(&sparse)))
    return rewriter.notifyMatchFailure(
        op, "only constant sparse is currently supported");
  if (sparse)
    return rewriter.notifyMatchFailure(
        op, "sparse gradients is currently not supported");

  Value output =
      gatherTensorAlongSingleAxis(rewriter, op, weight, adaptor.indices(), 0);
  rewriter.replaceOpWithNewOp<mhlo::ConvertOp>(
      op, getTypeConverter()->convertType(op.getType()), output);

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenIndexSelectOp>::matchAndRewrite(
    AtenIndexSelectOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto self = adaptor.self();
  auto selfTy = self.getType().template cast<RankedTensorType>();
  if (!selfTy)
    return op.emitError("only ranked tensor types are supported");
  int64_t dim;
  if (!matchPattern(op.dim(), m_TorchConstantInt(&dim)))
    return rewriter.notifyMatchFailure(
        op, "only constant dim is currently supported");

  Value output =
      gatherTensorAlongSingleAxis(rewriter, op, self, adaptor.index(), dim);

  rewriter.replaceOpWithNewOp<mhlo::ConvertOp>(
      op, getTypeConverter()->convertType(op.getType()), output);

  return success();
}

void populateTorchMlirReductionPatternsAndLegality(
    TypeConverter& typeConverter,
    RewritePatternSet& patterns,
    ConversionTarget& target) {
  MLIRContext* context = patterns.getContext();

#define INSERT_ATENOP_PATTERN(AtenOp) \
  target.addIllegalOp<AtenOp>();      \
  patterns.add<ConvertAtenOp<AtenOp>>(typeConverter, context);
  INSERT_ATENOP_PATTERN(AtenEmbeddingOp);
  INSERT_ATENOP_PATTERN(AtenIndexSelectOp);
  INSERT_ATENOP_PATTERN(AtenMaxDimOp);
#undef INSERT_ATENOP_PATTERN
}

} // namespace TorchConversion
} // namespace torch
} // namespace mlir
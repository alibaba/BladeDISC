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

#include "torch-mlir/Conversion/TorchToMhlo/TorchToMhlo.h"
#include "torch-mlir/Conversion/TorchToMhlo/MhloLegalizeUtils.h"

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
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::TorchConversion;

namespace {

// These legalizations are for unary ops with only for floating point datatypes.
// There is no supported quantized integer mode for these.
template <typename AtenOpT, typename MhloOpT>
class ConvertAtenUnaryFPOnlyOp : public OpConversionPattern<AtenOpT> {
 public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult matchAndRewrite(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value self = adaptor.self();
    auto selfTy = self.getType().cast<TensorType>();

    if (!selfTy)
      return op.emitError("Only Tensor types supported in MHLO");

    if (selfTy.getElementType().isa<mlir::FloatType>()) {
      rewriter.replaceOpWithNewOp<MhloOpT>(
          op,
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              op.getType()),
          self);
      return success();
    } else {
      return op.emitError(
          "Only floating-point datatype legalization supported");
    }
  }
};

// These unary op legalizations are identical for floating-point
// or quantized types
template <typename AtenOpT, typename MhloOpT>
class ConvertAtenUnaryOp : public OpConversionPattern<AtenOpT> {
 public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult matchAndRewrite(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<MhloOpT>(
        op,
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()),
        adaptor.self());
    return success();
  }
};

// These binary op legalizations are identical for floating-point
// or quantized types
template <typename AtenOpT, typename MhloOpT>
class ConvertAtenBinaryOp : public OpConversionPattern<AtenOpT> {
 public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult matchAndRewrite(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value lhs = adaptor.self();
    auto lhsTy = lhs.getType().cast<TensorType>();
    Value rhs = adaptor.other();
    auto rhsTy = rhs.getType().cast<TensorType>();

    if (!lhsTy || !rhsTy)
      return op.emitError("Only Tensor types supported in MHLO");

    auto lhsElemTy = lhsTy.getElementType();
    auto rhsElemTy = rhsTy.getElementType();

    if (lhsElemTy != rhsElemTy)
      return op.emitError("Add: input datatypes mismatched");

    rewriter.replaceOpWithNewOp<MhloOpT>(
        op,
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()),
        lhs,
        rhs);
    return success();
  }
};

template <typename T>
static bool isInValidRange(
    bool isFloat,
    const double& doubleValue,
    bool isInt,
    const int64_t& intValue) {
  if (isFloat) {
    // Do a round-trip check here instead of numeric limits due to
    // compiler warnings around double <-> int conversion.
    return (doubleValue == static_cast<double>(static_cast<T>(doubleValue)));
  } else {
    assert(isInt);
    return (intValue >= std::numeric_limits<T>::min()) &&
        (intValue <= std::numeric_limits<T>::max());
  }
  return true;
}

// Returns 1D 64-bit dense elements attribute with the given values.
inline mlir::DenseIntElementsAttr BuildI64ElementsAttr(
    mlir::OpBuilder& builder,
    const mlir::ArrayRef<int64_t>& values) {
  mlir::RankedTensorType ty = mlir::RankedTensorType::get(
      {static_cast<int64_t>(values.size())}, builder.getIntegerType(64));
  return mlir::DenseIntElementsAttr::get(ty, values);
}

inline SmallVector<int64_t, 4> RangeIndices(int64_t min, int64_t max) {
  SmallVector<int64_t, 4> range;
  for (int64_t k = min; k < max; ++k) {
    range.push_back(k);
  }
  return range;
}

// FIXME: This will eventually go into a Mhlo*Utils file.
LogicalResult torchScalarToMhloTensor(
    ConversionPatternRewriter& rewriter,
    Operation* op,
    Value torchScalarValue,
    Value& mhloTensor,
    Type dtype,
    llvm::ArrayRef<int64_t> dshape) {
  // Retrieve a const float or int value but create the out Tensor with dtype.
  double doubleValue;
  auto isFloat =
      matchPattern(torchScalarValue, m_TorchConstantFloat(&doubleValue));

  int64_t intValue;
  auto isInt = matchPattern(torchScalarValue, m_TorchConstantInt(&intValue));

  if (!isFloat && !isInt)
    return op->emitError("Unable to extract the scalar constant");

  if (dtype.isa<mlir::FloatType>()) {
    mhloTensor = mhlo::getConstTensor<float>(
                     rewriter, op, (isFloat ? doubleValue : intValue), dshape)
                     .getValue();
  } else if (auto intType = dtype.dyn_cast<mlir::IntegerType>()) {
    auto w = intType.getWidth();
    if (w != 32 && w != 64)
      return op->emitError("Unsupported integer type") << intType;

    if (w == 32) {
      if (!isInValidRange<int32_t>(isFloat, doubleValue, isInt, intValue)) {
        return op->emitError(
            "Supplied value of scalar constant exceeds limits "
            "of destination type");
      }
      int32_t d = isFloat ? static_cast<int32_t>(doubleValue)
                          : static_cast<int32_t>(intValue);
      mhloTensor =
          mhlo::getConstTensor<int32_t>(rewriter, op, {d}, dshape).getValue();
    } else if (w == 64) {
      if (!isInValidRange<int64_t>(isFloat, doubleValue, isInt, intValue)) {
        return op->emitError(
            "Supplied value of scalar constant exceeds limits "
            "of destination type");
      }
      int64_t d = (isFloat ? static_cast<int64_t>(doubleValue) : intValue);
      mhloTensor =
          mhlo::getConstTensor<int64_t>(rewriter, op, {d}, dshape).getValue();
    }
  } else
    return op->emitError("Usupported element type");

  return success();
}

LogicalResult torchAlphaToMhloTensor(
    ConversionPatternRewriter& rewriter,
    Operation* op,
    Value alphaScalar,
    Value& alphaTensor,
    Type dtype,
    bool checkForUnity) {
  if (succeeded(torchScalarToMhloTensor(
          rewriter, op, alphaScalar, alphaTensor, dtype, {})))
    return success();

  // `alpha` has not been specified.
  int64_t alphaValue;
  if (!matchPattern(alphaScalar, m_TorchConstantInt(&alphaValue)))
    return op->emitError(
        "Currently only scalar constants are supported for "
        "alpha in MHLO operation");
  // When no alpha has been specified, this must be 1.
  if (checkForUnity && alphaValue != 1)
    return op->emitError("Unsupported integer value for alpha");

  alphaTensor =
      mlir::mhlo::getMhloConstTensorSingleF32(rewriter, op, alphaValue);

  return success();
}

// These binary op legalizations are specific to add/sub which have an
// alpha multiplier.
template <typename AtenOpT, typename MhloOpT>
class ConvertAtenAddSubOp : public OpConversionPattern<AtenOpT> {
 public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult matchAndRewrite(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value lhs = adaptor.self();
    auto lhsType = lhs.getType().dyn_cast<TensorType>();
    Value rhs = adaptor.other();
    auto rhsType = rhs.getType().dyn_cast<TensorType>();

    if (!lhsType)
      return op.emitError("Only Tensor types supported in MHLO");

    if (auto lhsElemTy = lhsType.getElementType().dyn_cast<IntegerType>()) {
      if (lhsElemTy.getWidth() > 32)
        return op.emitError(
            "Integers with widths greater than 32 are not supported");
    }

    auto outType = OpConversionPattern<AtenOpT>::getTypeConverter()
                       ->convertType(op.getType())
                       .template cast<TensorType>();

    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat()) {
      return op.emitError(
          "Only floating-point or integer datatype legalization supported");
    }

    Value rhsAsTensor;
    if (!rhsType) {
      if (failed(torchScalarToMhloTensor(
              rewriter, op, op.other(), rhsAsTensor, outElemTy, {})))
        return op.emitError(
            "Currently only scalar constants are supported for "
            "conversion in MHLO operation");
    }
    auto rhsTensor = rhsType ? rhs : rhsAsTensor;

    // Handle alpha.
    Value alphaTensor;
    if (failed(torchAlphaToMhloTensor(
            rewriter,
            op.getOperation(),
            op.alpha(),
            alphaTensor,
            outElemTy,
            /*checkForUnity=*/false))) {
      return op.emitError(
          "Currently only scalar constants are supported for "
          "alpha in conversion to MHLO operation");
    }

    auto mhlo_shape =
        mhlo::getMhloShapeOfTensor(rewriter, op.getOperation(), rhsTensor);
    auto broadcast_dims = DenseIntElementsAttr::get(
        RankedTensorType::get({0}, rewriter.getI64Type()), ArrayRef<int64_t>{});
    auto alpha_broadcast = rewriter
                               .create<mhlo::DynamicBroadcastInDimOp>(
                                   op.getLoc(),
                                   rhsTensor.getType(),
                                   alphaTensor,
                                   mhlo_shape,
                                   broadcast_dims)
                               .getResult();
    auto multTensor = rewriter.create<mhlo::MulOp>(
        op.getLoc(),
        rhsType ? rhsType : RankedTensorType::get({}, outElemTy),
        rhsTensor,
        alpha_broadcast);

    if (outElemTy.isa<mlir::FloatType>()) {
      if (lhsType.getElementType() != outElemTy)
        lhs = rewriter.create<mhlo::ConvertOp>(op.getLoc(), outType, lhs);

      rewriter.replaceOpWithNewOp<MhloOpT>(
          op, outType, lhs, multTensor, nullptr);

      return success();
    } else {
      return op.emitError(
          "Only floating-point datatype legalization supported");
    }
  }
}; // namespace

// Binary op legalizations for comparator ops.
template <typename AtenOpT, typename MhloOpT>
class ConvertAtenCompareOp : public OpConversionPattern<AtenOpT> {
 public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult matchAndRewrite(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value lhs = adaptor.self();
    auto lhsTy = lhs.getType().dyn_cast<TensorType>();
    Value rhs = adaptor.other();
    auto rhsTy = rhs.getType().dyn_cast<TensorType>();

    if (!lhsTy)
      return op.emitError("Only Tensor types supported in MHLO");

    auto lhsElemTy = lhsTy.getElementType();
    if (!lhsElemTy.isIntOrFloat())
      return op.emitError(
          "Only floating-point or integer datatype legalization supported");

    // For bitwise operators, only integer datatype legalization is supported
    if (lhsElemTy.isa<mlir::FloatType>() &&
        std::is_same<AtenOpT, AtenBitwiseAndTensorOp>()) {
      return op.emitError(
          "For bitwise operators, only integer datatype "
          "legalization is supported");
    }

    Value rhsAsTensor;
    if (!rhsTy) {
      if (failed(torchScalarToMhloTensor(
              rewriter, op, op.other(), rhsAsTensor, lhsElemTy, {})))
        return op.emitError(
            "Currently only scalar constants are supported for "
            "conversion in MHLO operation");
    }
    auto rhsTensor = rhsTy ? rhs : rhsAsTensor;
    // There is no Lesser operator in MHLO.
    auto swapLhsRhs =
        (std::is_same<AtenOpT, AtenLtTensorOp>() ||
         std::is_same<AtenOpT, AtenLtScalarOp>());

    auto resultOp = rewriter.create<MhloOpT>(
        op.getLoc(),
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()),
        (swapLhsRhs ? rhsTensor : lhs),
        (swapLhsRhs ? lhs : rhsTensor));

    // There is no NE operator in MHLO.
    if (std::is_same<AtenOpT, AtenNeTensorOp>() ||
        std::is_same<AtenOpT, AtenNeScalarOp>())
      rewriter.replaceOpWithNewOp<mhlo::NotOp>(
          op,
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              op.getType()),
          resultOp.getResult());
    else
      rewriter.replaceOp(op, resultOp.getResult());

    return success();
  }
};

// Binary op legalizations for Mul variants.
template <typename AtenOpT>
class ConvertAtenMulOp : public OpConversionPattern<AtenOpT> {
 public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult matchAndRewrite(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value lhs = adaptor.self();
    auto lhsType = lhs.getType().dyn_cast<TensorType>();

    if (!lhsType)
      return op.emitError("Only Tensor types supported in MHLO");

    auto outType = OpConversionPattern<AtenOpT>::getTypeConverter()
                       ->convertType(op.getType())
                       .template cast<TensorType>();

    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat())
      return op.emitError(
          "Only floating-point or integer datatype legalization supported");

    Value rhsTensor;
    if (std::is_same<AtenOpT, AtenSquareOp>()) {
      rhsTensor = lhs;
    } else {
      Value rhsAsTensor;
      Value rhs = adaptor.other();
      auto rhsType = rhs.getType().dyn_cast<TensorType>();
      if (!rhsType) {
        if (failed(torchScalarToMhloTensor(
                rewriter, op, op.other(), rhsAsTensor, outElemTy, {})))
          return op.emitError(
              "Currently only scalar constants are supported for "
              "conversion in MHLO operation");
      }
      rhsTensor = rhsType ? rhs : rhsAsTensor;
    }

    if (outElemTy.isa<mlir::FloatType>() ||
        outElemTy.isa<mlir::IntegerType>()) {
      if (lhsType.getElementType() != outElemTy)
        lhs = rewriter.create<mhlo::ConvertOp>(op.getLoc(), outType, lhs);

      rewriter.replaceOpWithNewOp<mhlo::MulOp>(
          op,
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              op.getType()),
          lhs,
          rhsTensor);
      return success();
    } else {
      // Quantized multiplication may need to rescale inputs.
      return op.emitError(
          "Only floating-point or integer datatype "
          "legalization currently supported");
    }
  }
};

template <typename AtenOpT>
class ConvertAtenDivOp : public OpConversionPattern<AtenOpT> {
 public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult matchAndRewrite(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value lhs = adaptor.self();
    auto lhsTy = lhs.getType().dyn_cast<TensorType>();
    Value rhs = adaptor.other();
    auto rhsTy = rhs.getType().dyn_cast<TensorType>();

    if (!lhsTy)
      return op.emitError("Only Tensor types supported in MHLO");

    auto lhsElemTy = lhsTy.getElementType();
    if (!lhsElemTy.isIntOrFloat())
      return op.emitError(
          "Only floating-point or integer datatype legalization supported");

    Value rhsAsTensor;
    if (!rhsTy) {
      if (failed(torchScalarToMhloTensor(
              rewriter, op, op.other(), rhsAsTensor, lhsElemTy, {})))
        return op.emitError(
            "Currently only scalar constants are supported for "
            "conversion in MHLO operation");
    }
    auto rhsTensor = rhsTy ? rhs : rhsAsTensor;

    rewriter.replaceOpWithNewOp<chlo::BroadcastDivOp>(
        op,
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()),
        lhs,
        rhsTensor,
        nullptr);
    return success();
  }
};

// This defines a template to construct ops whose legalizations are
// specialized.
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

template <>
LogicalResult ConvertAtenOp<AtenTanhOp>::matchAndRewrite(
    AtenTanhOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  Value self = adaptor.self();
  auto selfTy = self.getType().cast<TensorType>();
  if (selfTy && selfTy.getElementType().isa<mlir::FloatType>()) {
    rewriter.replaceOpWithNewOp<mhlo::TanhOp>(
        op, getTypeConverter()->convertType(op.getType()), self);
    return success();
  } else {
    // Sigmoid legalization in MHLO for quantized element-type uses
    // specialized mhlo.table construct.
    return op.emitError(
        "Only floating-point datatype legalization currently supported");
  }
}

template <>
LogicalResult ConvertAtenOp<AtenSigmoidOp>::matchAndRewrite(
    AtenSigmoidOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  Value self = adaptor.self();
  auto selfTy = self.getType().cast<TensorType>();
  if (selfTy && selfTy.getElementType().isa<mlir::FloatType>()) {
    // rewriter.replaceOpWithNewOp<mhlo::SigmoidOp>(
    //     op, getTypeConverter()->convertType(op.getType()), self);
    // return success();
    return op.emitError(
        "Only floating-point datatype legalization currently supported");
  } else {
    // Sigmoid legalization in MHLO for quantized element-type uses
    // specialized mhlo.table construct.
    return op.emitError(
        "Only floating-point datatype legalization currently supported");
  }
}

using ReductionConvFunc = llvm::Optional<Value> (*)(
    PatternRewriter&,
    Operation*,
    RankedTensorType,
    Value,
    ElementsAttr,
    bool);

// They all constitute a common form invoking the appropriate
// converion function in MhloLegalizeCommon.cpp
template <typename AtenOpT, ReductionConvFunc ConversionFuncT>
class ConvertAtenReductionOp : public OpConversionPattern<AtenOpT> {
 public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;

  // Each variant must implement corresponding parameter parsing options
  virtual LogicalResult readReduceDimsAndKeepDims(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter,
      ElementsAttr& reduceDimsAttr,
      bool& keepDims) const {
    return rewriter.notifyMatchFailure(
        op, "Unimplemented reduce_dims and keep_dims parsing function");
  }

  // Common rewriter for all reduction ops, calls the specific implementation of
  // readReduceDimsAndKeepDims() needed for the op variant.
  LogicalResult matchAndRewrite(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value self = adaptor.self();
    auto selfTy = self.getType().cast<TensorType>();

    if (!selfTy)
      return op.emitError("Only Tensor types supported in MHLO");

    auto outputTy = OpConversionPattern<AtenOpT>::getTypeConverter()
                        ->convertType(op.getType())
                        .template cast<RankedTensorType>();
    if (!outputTy)
      return op.emitError(
          "Only ranked tensor type outputs permitted for reduce_mean");

    ElementsAttr reduceDimsAttr;
    bool keepDims;

    if (failed(readReduceDimsAndKeepDims(
            op, adaptor, rewriter, reduceDimsAttr, keepDims)))
      return failure();

    llvm::Optional<Value> result =
        ConversionFuncT(rewriter, op, outputTy, self, reduceDimsAttr, keepDims);

    if (!result)
      return failure();

    // TBD - support dtype casting.

    rewriter.replaceOp(op, {result.getValue()});

    return success();
  }
};

// This reduction op legalization template handles op variants that have
// explicit reduce_dims dimensions (provided as a list) and keep_dims
// parameters.
template <typename AtenOpT, ReductionConvFunc ConversionFuncT>
class ConvertAtenMultipleDimsReductionOp
    : public ConvertAtenReductionOp<AtenOpT, ConversionFuncT> {
  using ConvertAtenReductionOp<AtenOpT, ConversionFuncT>::
      ConvertAtenReductionOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult readReduceDimsAndKeepDims(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter,
      ElementsAttr& reduceDimsAttr,
      bool& keepDims) const override {
    SmallVector<int64_t, 4> reduceDims;
    if (!matchPattern(op.dim(), m_TorchConstantIntList(reduceDims)))
      return rewriter.notifyMatchFailure(
          op, "non-const dim parameter unsupported");
    int64_t N = reduceDims.size();
    auto reduceDimsType = RankedTensorType::get({N}, rewriter.getI64Type());
    reduceDimsAttr = DenseIntElementsAttr::get(
        reduceDimsType, llvm::makeArrayRef(reduceDims));

    keepDims = false;
    if (!matchPattern(op.keepdim(), m_TorchConstantBool(&keepDims)))
      return rewriter.notifyMatchFailure(
          op, "non-const keepdim parameter unsupported");

    return success();
  }
};

// This reduction op legalization template handles op variants that reduce in
// only one explicit dim which is provided as a number (rather than a list), and
// a keep_dims parameter.
template <typename AtenOpT, ReductionConvFunc ConversionFuncT>
class ConvertAtenOneDimReductionOp
    : public ConvertAtenReductionOp<AtenOpT, ConversionFuncT> {
  using ConvertAtenReductionOp<AtenOpT, ConversionFuncT>::
      ConvertAtenReductionOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult readReduceDimsAndKeepDims(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter,
      ElementsAttr& reduceDimsAttr,
      bool& keepDims) const override {
    int64_t reduceDim;
    if (!matchPattern(op.dim(), m_TorchConstantInt(&reduceDim)))
      return rewriter.notifyMatchFailure(
          op, "non-const dim parameter unsupported");
    auto reduceDimsType = RankedTensorType::get({1}, rewriter.getI64Type());
    reduceDimsAttr = DenseIntElementsAttr::get(
        reduceDimsType, llvm::makeArrayRef({reduceDim}));

    keepDims = false;
    if (!matchPattern(op.keepdim(), m_TorchConstantBool(&keepDims)))
      return rewriter.notifyMatchFailure(
          op, "non-const keepdim parameter unsupported");

    return success();
  }
};

// This reduction op legalization template handles op variants that reduce all
// dims does not keep dims.
template <typename AtenOpT, ReductionConvFunc ConversionFuncT>
class ConvertAtenAllDimsReductionOp
    : public ConvertAtenReductionOp<AtenOpT, ConversionFuncT> {
 public:
  using ConvertAtenReductionOp<AtenOpT, ConversionFuncT>::
      ConvertAtenReductionOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult readReduceDimsAndKeepDims(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter,
      ElementsAttr& reduceDimsAttr,
      bool& keepDims) const override {
    auto self = adaptor.self();
    auto selfTy = self.getType().template cast<RankedTensorType>();

    // Select all dims to reduce
    SmallVector<int64_t, 4> reduceDims;
    for (int64_t i = 0; i < selfTy.getRank(); i++)
      reduceDims.push_back(i);
    int64_t N = selfTy.getRank();
    auto reduceDimsType = RankedTensorType::get({N}, rewriter.getI64Type());
    reduceDimsAttr = DenseIntElementsAttr::get(
        reduceDimsType, llvm::makeArrayRef(reduceDims));
    keepDims = false;

    return success();
  }
};

template <>
LogicalResult ConvertAtenOp<AtenPowTensorScalarOp>::matchAndRewrite(
    AtenPowTensorScalarOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  Value self = adaptor.self();
  auto selfTy = self.getType().template cast<RankedTensorType>();

  if (!selfTy)
    return op.emitError("Only ranked tensor types supported in MHLO Pow");

  if (!selfTy.getElementType().isa<mlir::FloatType>())
    return op.emitError("Only floating-point datatype legalization supported");

  Value expTensor;
  Value expScalar = op.exponent();
  if (failed(torchScalarToMhloTensor(
          rewriter, op, expScalar, expTensor, selfTy.getElementType(), {})))
    return op.emitError(
        "Currently only scalar constants are supported for "
        "conversion in MHLO Pow operation");

  rewriter.replaceOpWithNewOp<mhlo::PowOp>(
      op, getTypeConverter()->convertType(op.getType()), self, expTensor);

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenRsubScalarOp>::matchAndRewrite(
    AtenRsubScalarOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto self = adaptor.self();
  auto otherScalar = op.other();
  auto alphaScalar = op.alpha();

  auto selfTy = self.getType().template cast<RankedTensorType>();
  if (!selfTy)
    return op.emitError("Only ranked tensor types supported in MHLO Rsub");

  if (!selfTy.getElementType().isa<mlir::FloatType>())
    return op.emitError("Only floating-point datatype legalization supported");

  Value otherTensor, alphaTensor;

  if (failed(torchScalarToMhloTensor(
          rewriter, op, otherScalar, otherTensor, selfTy.getElementType(), {})))
    return op.emitError(
        "Currently only scalar constants are supported for "
        "conversion in MHLO Rsub operation");

  if (failed(torchAlphaToMhloTensor(
          rewriter,
          op.getOperation(),
          alphaScalar,
          alphaTensor,
          selfTy.getElementType(),
          /*checkForUnity=*/true)))
    return failure();

  auto multTensor = rewriter.create<mhlo::MulOp>(
      op->getLoc(),
      getTypeConverter()->convertType(op.getType()),
      self,
      alphaTensor);

  rewriter.replaceOpWithNewOp<mhlo::SubOp>(
      op,
      getTypeConverter()->convertType(op.getType()),
      otherTensor,
      multTensor);

  return success();
}

// Torch constants are converted to mhlo.const .
template <>
LogicalResult ConvertAtenOp<ValueTensorLiteralOp>::matchAndRewrite(
    ValueTensorLiteralOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto outputTy = getTypeConverter()
                      ->convertType(op.getType())
                      .template cast<RankedTensorType>();
  rewriter.replaceOpWithNewOp<mhlo::ConstOp>(op, outputTy, adaptor.value());

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenLog2Op>::matchAndRewrite(
    AtenLog2Op op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  // Not a tensor type.
  auto selfType = adaptor.self().getType().dyn_cast<TensorType>();
  if (!selfType)
    return op.emitError("Only tensor types are currently supported");

  // Constant value of ln2.
  SmallVector<int64_t> ln2Shape(selfType.getRank(), 1);
  auto ln2Op =
      mhlo::getConstTensor<float>(rewriter, op, {0.69314718056}, ln2Shape)
          .getValue();
  auto one =
      mhlo::getConstTensor<float>(rewriter, op.getOperation(), {1.0}, {1})
          .getValue();
  auto rcpOp =
      rewriter.create<mhlo::DivOp>(op.getLoc(), ln2Op.getType(), one, ln2Op);

  auto outType = getTypeConverter()->convertType(op.getType());
  auto logOp =
      rewriter.create<mhlo::LogOp>(op.getLoc(), outType, adaptor.self());
  rewriter.replaceOpWithNewOp<mhlo::MulOp>(op, outType, logOp, rcpOp);

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenSizeIntOp>::matchAndRewrite(
    AtenSizeIntOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  // Not a tensor type.
  auto selfType = adaptor.self().getType().dyn_cast<TensorType>();
  if (!selfType)
    return op.emitError("Only tensor types are currently supported");
  auto dim = rewriter.create<arith::IndexCastOp>(
      op.getLoc(), rewriter.getIndexType(), adaptor.dim());
  auto dimSize = rewriter.create<tensor::DimOp>(
      op.getLoc(), rewriter.getIndexType(), adaptor.self(), dim);

  rewriter.replaceOpWithNewOp<arith::IndexCastOp>(
      op, getTypeConverter()->convertType(op.getType()), dimSize);

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenDropoutOp>::matchAndRewrite(
    AtenDropoutOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  // Not a tensor type.
  auto selfType = adaptor.input().getType().dyn_cast<TensorType>();
  if (!selfType)
    return op.emitError("Only tensor types are currently supported");

  // FIXME: train and p are not handled.

  bool train;
  if (!matchPattern(op.train(), m_TorchConstantBool(&train)))
    op.emitError("train must be a Scalar constant");

  if (train)
    op.emitError("train must be false");

  rewriter.replaceOpWithNewOp<mhlo::ConvertOp>(
      op, getTypeConverter()->convertType(op.getType()), adaptor.input());

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenViewOp>::matchAndRewrite(
    AtenViewOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  // Not a tensor type.
  auto selfType = adaptor.self().getType().dyn_cast<TensorType>();
  if (!selfType)
    return op.emitError("Only tensor types are currently supported");

  SmallVector<Value> dimsSize;
  if (!getListConstructElements(adaptor.size(), dimsSize)) {
    return op.emitError("Dims size must be a list of Scalar");
  }

  auto loc = op.getLoc();
  auto rankType = selfType.dyn_cast<RankedTensorType>();
  auto newRank = dimsSize.size();
  for (size_t d = 0; d < newRank; ++d) {
    auto dsize = dimsSize[d];
    int64_t dval;
    if (matchPattern(dsize, m_TorchConstantInt(&dval)) && dval == -1) {
      return op.emitError("The size cannot be set to -1.");
    } else {
      dsize = rewriter.create<ToI64Op>(loc, dsize).getResult();
      dsize = rewriter.create<mlir::arith::IndexCastOp>(
          loc, rewriter.getIndexType(), dsize);
    }
    dsize = rewriter.create<mlir::arith::IndexCastOp>(
        loc, rewriter.getI32Type(), dsize);
    dimsSize[d] = dsize;
  }

  auto mhloShape = rewriter.create<mlir::tensor::FromElementsOp>(loc, dimsSize);

  rewriter.replaceOpWithNewOp<mhlo::DynamicReshapeOp>(
      op,
      getTypeConverter()->convertType(op.getType()),
      adaptor.self(),
      mhloShape);

  return success();
}

// Ref: https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html
// aten.broadcast_to has similar semantics with torch.expand
template <>
LogicalResult ConvertAtenOp<AtenBroadcastToOp>::matchAndRewrite(
    AtenBroadcastToOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  // Not a tensor type.
  //
  auto selfType = adaptor.self().getType().dyn_cast<TensorType>();
  if (!selfType)
    return op.emitError("Only tensor types are currently supported");

  SmallVector<Value> dimsSize;
  if (!getListConstructElements(adaptor.size(), dimsSize)) {
    return op.emitError("Dims size must be a list of Scalar");
  }

  auto loc = op.getLoc();
  auto rankType = selfType.dyn_cast<RankedTensorType>();
  auto selfRank = rankType ? rankType.getRank() : 0;
  auto newRank = dimsSize.size();
  auto leadingRank = newRank - selfRank;
  for (size_t d = 0; d < newRank; ++d) {
    // !torch.int
    auto dsize = dimsSize[d];
    int64_t dval;
    if (matchPattern(dsize, m_TorchConstantInt(&dval)) && dval == -1) {
      if (d < leadingRank) {
        return op.emitError(
            "Ref: https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html."
            "For the new leading dimensions, the size cannot be set to -1.");
      } else {
        // Passing -1 as the size for a dimension means not changing the size
        // of that dimension.
        //
        // tensor.dim %self -> index
        dsize = rewriter.create<tensor::DimOp>(
            loc, adaptor.self(), d - leadingRank);
      }
    } else {
      // !torch.int -> i64
      dsize = rewriter.create<ToI64Op>(loc, dsize).getResult();
      // i64 -> index
      dsize = rewriter.create<mlir::arith::IndexCastOp>(
          loc, rewriter.getIndexType(), dsize);
    }
    // index -> i32
    dsize = rewriter.create<mlir::arith::IndexCastOp>(
        loc, rewriter.getI32Type(), dsize);
    dimsSize[d] = dsize;
  }

  auto mhloShape = rewriter.create<mlir::tensor::FromElementsOp>(loc, dimsSize);
  auto broadcastDims =
      BuildI64ElementsAttr(rewriter, RangeIndices(leadingRank, newRank));
  rewriter.replaceOpWithNewOp<mhlo::DynamicBroadcastInDimOp>(
      op,
      getTypeConverter()->convertType(op.getType()),
      adaptor.self(),
      mhloShape,
      broadcastDims);
  return success();
}

template <typename AtenOpT, typename MhloOpT>
class ConvertAtenPoolingBaseOp : public OpConversionPattern<AtenOpT> {
 public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;

  // Different pooling variants need to process inputs differently, e.g.
  // adaptive pooling generates the kernel size rather than receive it. This
  // function also transposes inputs.
  virtual LogicalResult processInputs(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter,
      Value& input,
      ArrayAttr& kernel,
      ArrayAttr& stride,
      ArrayAttr& pad,
      Type& outputTy) const {
    return rewriter.notifyMatchFailure(
        op, "Unimplemented pooling input parsing function");
  }

  int64_t getOutputDim(
      int64_t inputDim,
      int64_t kernelDim,
      int64_t stride,
      int64_t padBefore,
      int64_t padAfter,
      int64_t dilation) const {
    if (inputDim == ShapedType::kDynamicSize) {
      return ShapedType::kDynamicSize;
    } else {
      return (
          (inputDim + padBefore + padAfter - dilation * (kernelDim - 1) - 1) /
              stride +
          1);
    }
  }

  // Apply the transposeDims vector on input to generate a transposed form.
  Value transposeTensor(
      AtenOpT op,
      ConversionPatternRewriter& rewriter,
      Value input,
      ArrayRef<int32_t> transposeDims) const {
    auto inputTy = input.getType().template cast<RankedTensorType>();
    auto inputElemTy = inputTy.getElementType();
    auto inputShape = inputTy.getShape();
    auto inputRank = inputTy.getRank();

    llvm::Optional<Value> transposeDimsConst = mhlo::getConstTensor<int32_t>(
        rewriter,
        op,
        /*vec=*/transposeDims,
        /*shape=*/{static_cast<int32_t>(inputRank)});

    SmallVector<int64_t> transposedInputShape;
    for (auto& dim : transposeDims)
      transposedInputShape.push_back(inputShape[dim]);
    auto transposedInputType =
        RankedTensorType::get(transposedInputShape, inputElemTy);
    return rewriter
        .create<mhlo::TransposeOp>(
            op->getLoc(),
            transposedInputType,
            input,
            transposeDimsConst.getValue())
        .getResult();
  }

  Value transposePoolingInputToHwc(
      AtenOpT op,
      ConversionPatternRewriter& rewriter,
      Value input) const {
    auto inputRank =
        input.getType().template cast<RankedTensorType>().getRank();

    SmallVector<int32_t> nchwToNhwc4DTransposeDims({0, 2, 3, 1});
    SmallVector<int32_t> chwToHwc3DTransposeDims({1, 2, 0});

    return transposeTensor(
        op,
        rewriter,
        input,
        inputRank == 3 ? chwToHwc3DTransposeDims : nchwToNhwc4DTransposeDims);
  }

  Value transposePoolingOutputToChw(
      AtenOpT op,
      ConversionPatternRewriter& rewriter,
      Value input) const {
    auto inputTy = input.getType().template cast<RankedTensorType>();
    auto inputRank = inputTy.getRank();

    SmallVector<int32_t> nhwcToNchw4DTransposeDims({0, 3, 1, 2});
    SmallVector<int32_t> hwcToChw3DTransposeDims({2, 0, 1});

    return transposeTensor(
        op,
        rewriter,
        input,
        inputRank == 3 ? hwcToChw3DTransposeDims : nhwcToNchw4DTransposeDims);
  }

  LogicalResult matchAndRewrite(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value input;
    ArrayAttr kernel, stride, pad;
    Type outputTy;

    // Attempts to read input and kernel parameters, or synthesize them in the
    // case of adaptive pooling. Also performs input CHW->HWC transpose.
    if (failed(processInputs(
            op, adaptor, rewriter, input, kernel, stride, pad, outputTy)))
      return op.emitError("Failed to process inputs for pooling");

    auto pooledOutput =
        rewriter
            .create<MhloOpT>(op->getLoc(), outputTy, input, kernel, stride, pad)
            .getResult();

    auto transposedOutput =
        ConvertAtenPoolingBaseOp<AtenOpT, MhloOpT>::transposePoolingOutputToChw(
            op, rewriter, pooledOutput);

    rewriter.replaceOpWithNewOp<mhlo::ConvertOp>(
        op,
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()),
        transposedOutput);

    return success();
  }
};

template <typename AtenOpT, typename MhloOpT>
class ConvertAtenAdaptivePoolingOp
    : public ConvertAtenPoolingBaseOp<AtenOpT, MhloOpT> {
 public:
  using ConvertAtenPoolingBaseOp<AtenOpT, MhloOpT>::ConvertAtenPoolingBaseOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult processInputs(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter,
      Value& input,
      ArrayAttr& kernel,
      ArrayAttr& stride,
      ArrayAttr& pad,
      Type& outputTy) const override {
    auto inputXchw = adaptor.self();
    auto inputTy = inputXchw.getType().template cast<RankedTensorType>();
    if (!inputTy)
      return op.emitError("Adaptive avgpool requires ranked tensor input");

    auto inputShape = inputTy.getShape();
    auto inputRank = inputTy.getRank();
    auto inputElemTy = inputTy.getElementType();

    // Rank sanity check.
    if (inputTy.getRank() != 4 && inputRank != 3)
      return op.emitError("NCHW->NHWC transpose requires 3D or 4D tensor");

    int64_t inputHDim = inputShape[inputRank - 2];
    int64_t inputWDim = inputShape[inputRank - 1];

    SmallVector<int64_t> outputSize;
    if (!matchPattern(op.output_size(), m_TorchConstantIntList(outputSize)))
      return rewriter.notifyMatchFailure(
          op, "Non-const output_size for adaptive pooling unsupported.");

    SmallVector<int64_t> kernelDims;
    int64_t outputHDim, outputWDim;
    if (outputSize.size() == 1) {
      outputHDim = outputWDim = outputSize[0];
    } else {
      if (outputSize.size() != 2)
        return op.emitError(
            "Adaptive avgpool output_size not 1 or 2 elements.");

      // Assumes 'None' (e.g. output_size=(None, 5) ) is expressed as <=0.
      outputHDim =
          (outputSize[0] <= 0) ? inputShape[inputRank - 2] : outputSize[0];
      outputWDim =
          (outputSize[1] <= 0) ? inputShape[inputRank - 1] : outputSize[1];
    }

    // In adaptive pooling,
    // stride = inputDim // outputDim
    // kernel = inputDim - (outputDim-1)* stride
    // pad = 0, dilation = 1

    int64_t strideH = inputShape[inputRank - 2] / outputHDim;
    int64_t strideW = inputShape[inputRank - 1] / outputWDim;

    kernelDims.push_back(inputHDim - (outputHDim - 1) * strideH);
    kernelDims.push_back(inputWDim - (outputWDim - 1) * strideW);

    SmallVector<int64_t> outputShape;
    if (inputRank > 3)
      outputShape.push_back(inputShape[0]);
    outputShape.push_back(outputHDim);
    outputShape.push_back(outputWDim);
    outputShape.push_back(inputShape[inputRank - 3]);

    // Transpose to xHWC
    input =
        ConvertAtenPoolingBaseOp<AtenOpT, MhloOpT>::transposePoolingInputToHwc(
            op, rewriter, inputXchw);
    kernel = rewriter.getI64ArrayAttr(kernelDims);
    stride = rewriter.getI64ArrayAttr({strideH, strideW});
    // Adaptive pooling does unit dilation and zero pad.
    pad = rewriter.getI64ArrayAttr({0, 0, 0, 0});
    outputTy = RankedTensorType::get(outputShape, inputElemTy);

    return success();
  }
};

template <typename AtenOpT, typename MhloOpT>
class ConvertAtenPoolingOp : public ConvertAtenPoolingBaseOp<AtenOpT, MhloOpT> {
 public:
  using ConvertAtenPoolingBaseOp<AtenOpT, MhloOpT>::ConvertAtenPoolingBaseOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult processInputs(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter,
      Value& input,
      ArrayAttr& kernel,
      ArrayAttr& stride,
      ArrayAttr& pad,
      Type& outputTy) const override {
    auto inputXchw = adaptor.self();
    auto inputTy = inputXchw.getType().template cast<RankedTensorType>();
    if (!inputTy)
      return op.emitError("Adaptive avgpool requires ranked tensor input");

    auto inputShape = inputTy.getShape();
    auto inputRank = inputTy.getRank();
    auto inputElemTy = inputTy.getElementType();

    // Rank sanity check.
    if (inputTy.getRank() != 4 && inputRank != 3)
      return op.emitError("NCHW->NHWC transpose requires 3D or 4D tensor");

    // Transpose to xHWC
    input =
        ConvertAtenPoolingBaseOp<AtenOpT, MhloOpT>::transposePoolingInputToHwc(
            op, rewriter, inputXchw);

    SmallVector<int64_t> kernelSize;
    if (!matchPattern(op.kernel_size(), m_TorchConstantIntList(kernelSize)))
      return rewriter.notifyMatchFailure(
          op, "Non-const kernel_size for adaptive pooling unsupported.");
    kernel = rewriter.getI64ArrayAttr(kernelSize);

    SmallVector<int64_t> strideArray;
    if (!matchPattern(op.stride(), m_TorchConstantIntList(strideArray)))
      return rewriter.notifyMatchFailure(
          op, "Non-const stride for adaptive pooling unsupported.");
    stride = rewriter.getI64ArrayAttr(strideArray);

    SmallVector<int64_t> padArray;
    if (!matchPattern(op.padding(), m_TorchConstantIntList(padArray)))
      return rewriter.notifyMatchFailure(
          op, "Non-const pad for adaptive pooling unsupported.");
    pad = rewriter.getI64ArrayAttr(
        {padArray[0], padArray[0], padArray[1], padArray[1]});

    SmallVector<int64_t> dilationArray;
    if (!matchPattern(op.dilation(), m_TorchConstantIntList(dilationArray)))
      return rewriter.notifyMatchFailure(
          op, "Non-const dilation for adaptive pooling unsupported.");
    // MHLO pooling only supports unit dilation.
    if (dilationArray[0] > 1 || dilationArray[1] > 1)
      return op.emitError("Cannot process non-unit pooling dilation.");

    // FIXME: add ceil_mode support.

    int64_t outputHDim =
        ConvertAtenPoolingBaseOp<AtenOpT, MhloOpT>::getOutputDim(
            inputShape[inputRank - 2],
            kernelSize[0],
            strideArray[0],
            padArray[0],
            padArray[0],
            dilationArray[0]);
    int64_t outputWDim =
        ConvertAtenPoolingBaseOp<AtenOpT, MhloOpT>::getOutputDim(
            inputShape[inputRank - 1],
            kernelSize[1],
            strideArray[1],
            padArray[1],
            padArray[1],
            dilationArray[1]);
    SmallVector<int64_t> outputShape;
    if (inputRank > 3)
      outputShape.push_back(inputShape[0]);
    outputShape.push_back(outputHDim);
    outputShape.push_back(outputWDim);
    outputShape.push_back(inputShape[inputRank - 3]);
    outputTy = RankedTensorType::get(outputShape, inputElemTy);

    return success();
  }
};

// Ref: Error checking based on the Torch to LinAlg lowering
template <typename AtenOpT, int fillVal>
class ConvertAtenConstPatternOp : public OpConversionPattern<AtenOpT> {
 public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult matchAndRewrite(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto outType = OpConversionPattern<AtenOpT>::getTypeConverter()
                       ->convertType(op.getType())
                       .template dyn_cast<TensorType>();

    if (!outType)
      return op.emitError("Only Tensor types supported in MHLO");

    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat())
      return op.emitError(
          "Only floating-point or integer datatype legalization supported");

    // FIXME: Handle layout, device and pin_memory. Assume dtype has been
    // processed to set output type correctly?
    if (!op.layout().getType().template isa<Torch::NoneType>())
      return op.emitError("Only default layout is supported");

    bool pinMemory;
    if (!op.pin_memory().getType().template isa<Torch::NoneType>() &&
        (!matchPattern(op.pin_memory(), m_TorchConstantBool(&pinMemory)) ||
         pinMemory)) {
      return op.emitError(
          "Unsupported pin_memory, should be either None or false");
    }

    SmallVector<int64_t> shape;
    if (!matchPattern(op.size(), m_TorchConstantIntList(shape))) {
      return op.emitError("Shape must be a list of Scalar constants");
    }

    int64_t size = 1;
    for (auto s : shape)
      size *= s;

    SmallVector<int32_t> values(size, fillVal);
    auto constOp =
        mhlo::getConstTensor<int32_t>(rewriter, op, values, shape).getValue();

    rewriter.replaceOpWithNewOp<mhlo::ConvertOp>(op, outType, constOp);

    return success();
  }
};

template <typename AtenOpT>
class ConvertAtenFillScalarOp : public OpConversionPattern<AtenOpT> {
 public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult matchAndRewrite(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto outType = OpConversionPattern<AtenOpT>::getTypeConverter()
                       ->convertType(op.getType())
                       .template dyn_cast<TensorType>();

    if (!outType || !outType.hasStaticShape())
      return op.emitError(
          "Only Tensor types with static shapes are currently supported");

    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat()) {
      return op.emitError(
          "Only floating-point or integer datatype legalization supported");
    }
    Value constOp;
    if (failed(torchScalarToMhloTensor(
            rewriter, op, op.value(), constOp, outElemTy, outType.getShape())))
      return op.emitError("Supplied value must be a Scalar constant");

    rewriter.replaceOpWithNewOp<mhlo::ConvertOp>(op, outType, constOp);

    return success();
  }
};

} // namespace

// -----------------------------------------------------------------------------
// TorchToMhlo Pass
// -----------------------------------------------------------------------------

namespace {
class ConvertTorchToMhlo
    : public TorchConversion::ConvertTorchToMhloBase<ConvertTorchToMhlo> {
 public:
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<chlo::ChloDialect>();
    registry.insert<mhlo::MhloDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<arith::ArithmeticDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<
        chlo::ChloDialect,
        mhlo::MhloDialect,
        tensor::TensorDialect,
        arith::ArithmeticDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);

#define INSERT_UNARY_FPONLY_PATTERN(AtenOp, MhloOp)       \
  target.addIllegalOp<AtenOp>();                          \
  patterns.add<ConvertAtenUnaryFPOnlyOp<AtenOp, MhloOp>>( \
      typeConverter, context);
    INSERT_UNARY_FPONLY_PATTERN(AtenLogOp, mhlo::LogOp)
    INSERT_UNARY_FPONLY_PATTERN(AtenExpOp, mhlo::ExpOp)
    INSERT_UNARY_FPONLY_PATTERN(AtenErfOp, chlo::ErfOp)
#undef INSERT_UNARY_FPONLY_PATTERN

#define INSERT_UNARY_PATTERN(AtenOp, MhloOp) \
  target.addIllegalOp<AtenOp>();             \
  patterns.add<ConvertAtenUnaryOp<AtenOp, MhloOp>>(typeConverter, context);
    INSERT_UNARY_PATTERN(AtenNegOp, mhlo::NegOp)
    INSERT_UNARY_PATTERN(AtenFloorOp, mhlo::FloorOp)
    INSERT_UNARY_PATTERN(AtenRsqrtOp, mhlo::RsqrtOp)
    INSERT_UNARY_PATTERN(AtenBitwiseNotOp, mhlo::NotOp)
    INSERT_UNARY_PATTERN(AtenCeilOp, mhlo::CeilOp)

    // It's tricky that ConvertOp will use type from the return,
    // but not from the operand here.
    INSERT_UNARY_PATTERN(AtenContiguousOp, mhlo::ConvertOp)
    INSERT_UNARY_PATTERN(AtenToDtypeOp, mhlo::ConvertOp);
    INSERT_UNARY_PATTERN(AtenTypeAsOp, mhlo::ConvertOp);
#undef INSERT_UNARY_PATTERN

#define INSERT_BINARY_PATTERN(AtenOp, MhloOp) \
  target.addIllegalOp<AtenOp>();              \
  patterns.add<ConvertAtenBinaryOp<AtenOp, MhloOp>>(typeConverter, context);
    INSERT_BINARY_PATTERN(AtenMaximumOp, mhlo::MaxOp)
    INSERT_BINARY_PATTERN(AtenMinimumOp, mhlo::MinOp)
#undef INSERT_BINARY_PATTERN

#define INSERT_BINARY_ADDSUB_PATTERN(AtenOp, MhloOp) \
  target.addIllegalOp<AtenOp>();                     \
  patterns.add<ConvertAtenAddSubOp<AtenOp, MhloOp>>(typeConverter, context);
    // INSERT_BINARY_ADDSUB_PATTERN(AtenAddTensorOp, mhlo::AddOp)
    // INSERT_BINARY_ADDSUB_PATTERN(AtenAddScalarOp, mhlo::AddOp)
    INSERT_BINARY_ADDSUB_PATTERN(AtenAddTensorOp, chlo::BroadcastAddOp)
    INSERT_BINARY_ADDSUB_PATTERN(AtenAddScalarOp, chlo::BroadcastAddOp)
    // INSERT_BINARY_ADDSUB_PATTERN(AtenSubTensorOp, mhlo::SubOp)
    // INSERT_BINARY_ADDSUB_PATTERN(AtenSubScalarOp, mhlo::SubOp)
    INSERT_BINARY_ADDSUB_PATTERN(AtenSubTensorOp, chlo::BroadcastSubOp)
    INSERT_BINARY_ADDSUB_PATTERN(AtenSubScalarOp, chlo::BroadcastSubOp)
#undef INSERT_BINARY_ADDSUB_PATTERN

#define INSERT_BINARY_MUL_PATTERN(AtenOp) \
  target.addIllegalOp<AtenOp>();          \
  patterns.add<ConvertAtenMulOp<AtenOp>>(typeConverter, context);
    INSERT_BINARY_MUL_PATTERN(AtenMulTensorOp);
    INSERT_BINARY_MUL_PATTERN(AtenMulScalarOp);
#undef INSERT_BINARY_MUL_PATTERN

#define INSERT_BINARY_DIV_PATTERN(AtenOp) \
  target.addIllegalOp<AtenOp>();          \
  patterns.add<ConvertAtenDivOp<AtenOp>>(typeConverter, context);
    INSERT_BINARY_DIV_PATTERN(AtenDivTensorOp);
    INSERT_BINARY_DIV_PATTERN(AtenDivScalarOp);
#undef INSERT_BINARY_DIV_PATTERN

#define INSERT_CONSTANT_FILL_PATTERN(AtenOp, fillVal)       \
  target.addIllegalOp<AtenOp>();                            \
  patterns.add<ConvertAtenConstPatternOp<AtenOp, fillVal>>( \
      typeConverter, context);
    INSERT_CONSTANT_FILL_PATTERN(AtenOnesOp, 1);
    INSERT_CONSTANT_FILL_PATTERN(AtenZerosOp, 0);
#undef INSERT_CONSTANT_FILL_PATTERN

#define INSERT_FILL_SCALAR_PATTERN(AtenOp) \
  target.addIllegalOp<AtenOp>();           \
  patterns.add<ConvertAtenFillScalarOp<AtenOp>>(typeConverter, context);
    INSERT_FILL_SCALAR_PATTERN(AtenFill_ScalarOp);
#undef INSERT_FILL_SCALAR_PATTERN

#define INSERT_ATENOP_PATTERN(AtenOp) \
  target.addIllegalOp<AtenOp>();      \
  patterns.add<ConvertAtenOp<AtenOp>>(typeConverter, context);
    INSERT_ATENOP_PATTERN(AtenBroadcastToOp);
    INSERT_ATENOP_PATTERN(AtenSigmoidOp);
    INSERT_ATENOP_PATTERN(AtenSizeIntOp);
    INSERT_ATENOP_PATTERN(AtenTanhOp);
    // INSERT_ATENOP_PATTERN(AtenReluOp);
    // INSERT_ATENOP_PATTERN(AtenArgmaxOp);
    INSERT_ATENOP_PATTERN(AtenPowTensorScalarOp);
    INSERT_ATENOP_PATTERN(AtenRsubScalarOp);
    INSERT_ATENOP_PATTERN(ValueTensorLiteralOp);
    // INSERT_ATENOP_PATTERN(AtenReshapeOp);
    // INSERT_ATENOP_PATTERN(AtenFlattenUsingIntsOp);
    // INSERT_ATENOP_PATTERN(AtenPermuteOp);
    INSERT_ATENOP_PATTERN(AtenLog2Op);
    // INSERT_ATENOP_PATTERN(AtenUnsqueezeOp);
    INSERT_ATENOP_PATTERN(AtenDropoutOp);
    INSERT_ATENOP_PATTERN(AtenViewOp);
    // INSERT_ATENOP_PATTERN(AtenGeluOp);
    // INSERT_ATENOP_PATTERN(AtenGeluBackwardOp);
#undef INSERT_ATENOP_PATTERN

    if (failed(applyPartialConversion(
            getOperation(), target, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::torch::TorchConversion::
    createConvertTorchToMhloPass() {
  return std::make_unique<ConvertTorchToMhlo>();
}

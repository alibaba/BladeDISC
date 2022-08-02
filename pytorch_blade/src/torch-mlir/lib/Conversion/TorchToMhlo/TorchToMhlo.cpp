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

torch_upstream::ScalarType getScalarTypeForType(Type type) {
  if (type.isa<Float32Type>())
    return torch_upstream::ScalarType::Float;
  if (type.isa<Float64Type>())
    return torch_upstream::ScalarType::Double;
  if (type.isSignlessInteger(64))
    return torch_upstream::ScalarType::Long;
  if (type.isSignlessInteger(32))
    return torch_upstream::ScalarType::Int;
  if (type.isSignlessInteger(1))
    return torch_upstream::ScalarType::Bool;
  if (type.isBF16())
    return torch_upstream::ScalarType::BFloat16;
  if (type.isF16())
    return torch_upstream::ScalarType::Half;
  if (type.isF64())
    return torch_upstream::ScalarType::Double;

  if (type.isa<Torch::FloatType>()) {
    return torch_upstream::ScalarType::Double;
  }
  if (type.isa<Torch::IntType>()) {
    return torch_upstream::ScalarType::Int;
  }
  if (type.isa<Torch::BoolType>()) {
    return torch_upstream::ScalarType::Bool;
  }

  llvm::report_fatal_error("unhandled type for getScalarTypeForType");
}

Type getTypeForScalarType(
    MLIRContext* context,
    torch_upstream::ScalarType dtypeInt) {
  switch (dtypeInt) {
    case torch_upstream::ScalarType::Float:
      return Float32Type::get(context);
    case torch_upstream::ScalarType::Double:
      return Float64Type::get(context);
    case torch_upstream::ScalarType::Long:
      return IntegerType::get(context, 64);
    case torch_upstream::ScalarType::Int:
      return IntegerType::get(context, 32);
    case torch_upstream::ScalarType::Bool:
      return IntegerType::get(context, 1);
    case torch_upstream::ScalarType::BFloat16:
      return mlir::FloatType::getBF16(context);
    default:
      llvm::report_fatal_error("unhandled type for getTypeForScalarType");
  }
}

template <typename AtenOpT, typename MhloOpT>
class ConvertAtenBinaryBroadcastOp : public OpConversionPattern<AtenOpT> {
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
      return op.emitError("Input datatypes mismatched");

    rewriter.replaceOpWithNewOp<MhloOpT>(
        op,
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()),
        lhs,
        rhs,
        /*broadcast_attr*/ nullptr);
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

Value scalarToMhloTensor(
    ConversionPatternRewriter& rewriter,
    Operation* op,
    Value scalarValue,
    Type dtype,
    llvm::ArrayRef<int64_t> dshape) {
  auto tensor = rewriter.create<tensor::FromElementsOp>(
      op->getLoc(), ArrayRef<Value>{scalarValue});
  auto dtype_tensor =
      rewriter.create<mhlo::ConvertOp>(op->getLoc(), tensor, dtype);
  return rewriter.create<mhlo::ReshapeOp>(
      op->getLoc(),
      RankedTensorType::get(mlir::ArrayRef<int64_t>{}, dtype),
      dtype_tensor);
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

  if (!isFloat && !isInt) {
    return op->emitError("Unable to extract the constant for a torch scalar");
  }

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

LogicalResult torchScalarToMhloTensorLike(
    ConversionPatternRewriter& rewriter,
    Operation* op,
    Value torchScalarValue,
    Value input,
    Value& mhloTensor) {
  // Retrieve a const float or int value but create the out Tensor with dtype.
  double doubleValue;
  auto isFloat =
      matchPattern(torchScalarValue, m_TorchConstantFloat(&doubleValue));

  int64_t intValue;
  auto isInt = matchPattern(torchScalarValue, m_TorchConstantInt(&intValue));

  if (!isFloat && !isInt)
    return op->emitError("Unable to extract the scalar constant");

  auto dtype = input.getType().dyn_cast<TensorType>().getElementType();
  auto loc = op->getLoc();
  if (dtype.isa<mlir::FloatType>()) {
    mhloTensor = chlo::getConstantLike(
        rewriter, loc, (isFloat ? doubleValue : intValue), input);
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
      mhloTensor = chlo::getConstantLike(rewriter, loc, d, input);
    } else if (w == 64) {
      if (!isInValidRange<int64_t>(isFloat, doubleValue, isInt, intValue)) {
        return op->emitError(
            "Supplied value of scalar constant exceeds limits "
            "of destination type");
      }
      int64_t d = (isFloat ? static_cast<int64_t>(doubleValue) : intValue);
      mhloTensor = chlo::getConstantLike(rewriter, loc, d, input);
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

template <typename AtenOpT, mhlo::ComparisonDirection DirectionT>
class ConvertAtenBinaryCompareOp : public OpConversionPattern<AtenOpT> {
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
    if (!rhsTy) {
      rhs = scalarToMhloTensor(rewriter, op, adaptor.other(), lhsElemTy, {});
      rhsTy = rhs.getType().cast<TensorType>();
    }

    auto rhsElemTy = rhsTy.getElementType();

    if (lhsElemTy != rhsElemTy) {
      auto lhsScalarType = getScalarTypeForType(lhsElemTy);
      auto rhsScalarType = getScalarTypeForType(rhsElemTy);
      auto promotedType =
          torch_upstream::promoteTypes(lhsScalarType, rhsScalarType);
      auto elemType = getTypeForScalarType(op.getContext(), promotedType);
      lhs = rewriter.create<mhlo::ConvertOp>(op.getLoc(), lhs, elemType);
      rhs = rewriter.create<mhlo::ConvertOp>(op.getLoc(), rhs, elemType);
    }

    auto result =
        rewriter
            .create<chlo::BroadcastCompareOp>(
                op.getLoc(), lhs, rhs, /*broadcast_attr*/ nullptr, DirectionT)
            .getResult();
    rewriter.replaceOp(op, result);
    return success();
  }
};

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
    auto outType = OpConversionPattern<AtenOpT>::getTypeConverter()
                       ->convertType(op.getType())
                       .template cast<TensorType>();

    Type outElemTy = outType.getElementType();
    Value rhsAsTensor;
    if (!rhsType) {
      rhsAsTensor =
          scalarToMhloTensor(rewriter, op, adaptor.other(), outElemTy, {});
    }
    auto rhsTensor = rhsType ? rhs : rhsAsTensor;

    rhsType = rhsTensor.getType().dyn_cast<TensorType>();
    auto alphaTensor =
        scalarToMhloTensor(rewriter, op, adaptor.alpha(), outElemTy, {});

    auto alphaType = alphaTensor.getType().template cast<TensorType>();
    if (alphaType.getElementType() != outElemTy)
      alphaTensor =
          rewriter.create<mhlo::ConvertOp>(op.getLoc(), alphaTensor, outElemTy);
    if (rhsType.getElementType() != outElemTy)
      rhsTensor =
          rewriter.create<mhlo::ConvertOp>(op.getLoc(), rhsTensor, outElemTy);

    auto mulTensor = rewriter
                         .create<chlo::BroadcastMulOp>(
                             op.getLoc(),
                             rhsTensor.getType(),
                             rhsTensor,
                             alphaTensor,
                             nullptr)
                         .getResult();

    if (lhsType.getElementType() != outElemTy)
      lhs = rewriter.create<mhlo::ConvertOp>(op.getLoc(), lhs, outElemTy);

    rewriter.replaceOpWithNewOp<MhloOpT>(op, outType, lhs, mulTensor, nullptr);
    return success();
  }
}; // namespace

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
        rhsAsTensor =
            scalarToMhloTensor(rewriter, op, adaptor.other(), outElemTy, {});
      }
      rhsTensor = rhsType ? rhs : rhsAsTensor;
    }
    auto rhsType = rhsTensor.getType().dyn_cast<TensorType>();

    if (outElemTy.isa<mlir::FloatType>() ||
        outElemTy.isa<mlir::IntegerType>()) {
      if (lhsType.getElementType() != outElemTy)
        lhs = rewriter.create<mhlo::ConvertOp>(op.getLoc(), lhs, outElemTy);
      if (rhsType.getElementType() != outElemTy)
        rhsTensor =
            rewriter.create<mhlo::ConvertOp>(op.getLoc(), rhsTensor, outElemTy);

      rewriter.replaceOpWithNewOp<chlo::BroadcastMulOp>(
          op,
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              op.getType()),
          lhs,
          rhsTensor,
          nullptr);

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
    if (!rhsTy) {
      rhs = scalarToMhloTensor(rewriter, op, adaptor.other(), lhsElemTy, {});
    }
    rhsTy = rhs.getType().dyn_cast<TensorType>();

    auto rhsElemTy = rhsTy.getElementType();
    if (lhsElemTy != rhsElemTy) {
      auto lhsScalarType = getScalarTypeForType(lhsElemTy);
      auto rhsScalarType = getScalarTypeForType(rhsElemTy);
      auto promotedType =
          torch_upstream::promoteTypes(lhsScalarType, rhsScalarType);
      auto elemType = getTypeForScalarType(op.getContext(), promotedType);
      lhs = rewriter.create<mhlo::ConvertOp>(op.getLoc(), lhs, elemType);
      rhs = rewriter.create<mhlo::ConvertOp>(op.getLoc(), rhs, elemType);
      lhsTy = lhs.getType().dyn_cast<TensorType>();
      rhsTy = rhs.getType().dyn_cast<TensorType>();
    }

    auto outType = OpConversionPattern<AtenOpT>::getTypeConverter()
                       ->convertType(op.getType())
                       .template cast<TensorType>();

    Type outElemTy = outType.getElementType();
    if (lhsTy.getElementType() != outElemTy)
      lhs = rewriter.create<mhlo::ConvertOp>(op.getLoc(), lhs, outElemTy);
    if (rhsTy.getElementType() != outElemTy)
      rhs = rewriter.create<mhlo::ConvertOp>(op.getLoc(), rhs, outElemTy);

    auto result = rewriter
                      .create<chlo::BroadcastDivOp>(
                          op.getLoc(), outType, lhs, rhs, nullptr)
                      .getResult();

    if (!isa<AtenDivTensorModeOp>(op)) {
      rewriter.replaceOp(op, result);

      return success();
    }

    AtenDivTensorModeOp divTensorModeOp =
        llvm::dyn_cast<AtenDivTensorModeOp>(op.getOperation());
    std::string roundingMode;
    if (!matchPattern(
            divTensorModeOp.rounding_mode(),
            m_TorchConstantStr(roundingMode))) {
      return op.emitError("only support constant str rounding mode");
    }

    auto loc = op.getLoc();
    if (roundingMode == "trunc") {
      // "trunc" - rounds the results of the division towards zero. Equivalent
      // to C-style integer division.
      auto sign = rewriter.create<mhlo::SignOp>(loc, result);
      auto abs = rewriter.create<mhlo::AbsOp>(loc, result);
      auto floor = rewriter.create<mhlo::FloorOp>(loc, abs);
      result = rewriter.create<mhlo::MulOp>(loc, sign, floor).getResult();
    }
    if (roundingMode == "floor") {
      // "floor" - rounds the results of the division down. Equivalent to
      // floor division in Python (the // operator)
      result = rewriter.create<mhlo::FloorOp>(loc, result).getResult();
    }

    rewriter.replaceOp(op, result);
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

// Convert a Aten::Relu to HLO
// Relu(x) = 0 if x < 0 else x
template <>
LogicalResult ConvertAtenOp<AtenReluOp>::matchAndRewrite(
    AtenReluOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  Location loc = op.getLoc();
  Value input = adaptor.self();
  auto inputTy = input.getType().template dyn_cast<RankedTensorType>();
  if (!inputTy) {
    return op.emitError("Only RankedTensorType is supported in Aten ReLU op.");
  }
  Value zero = chlo::getConstantLike(rewriter, loc, 0.0, input);
  Value compareGtZero = rewriter.create<mhlo::CompareOp>(
      loc, input, zero, mhlo::ComparisonDirection::GT);
  rewriter.replaceOpWithNewOp<mhlo::SelectOp>(
      op, inputTy, compareGtZero, input, zero);
  return success();
}

// Convert a Aten::Relu6 to HLO
// Relu6(x) = min(AtenRelu(x), 6)
template <>
LogicalResult ConvertAtenOp<AtenRelu6Op>::matchAndRewrite(
    AtenRelu6Op op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  Location loc = op.getLoc();
  Value input = adaptor.self();
  auto inputTy = input.getType().template dyn_cast<RankedTensorType>();
  if (!inputTy) {
    return op.emitError("Only RankedTensorType is supported in Aten ReLU op.");
  }
  Value relu = rewriter.create<AtenReluOp>(loc, inputTy, input);
  Value six = chlo::getConstantLike(rewriter, loc, 6.0, input);
  Value compareLtSix = rewriter.create<mhlo::CompareOp>(
      loc, input, six, mhlo::ComparisonDirection::LT);
  rewriter.replaceOpWithNewOp<mhlo::SelectOp>(
      op, inputTy, compareLtSix, relu, six);
  return success();
}

// Convert a Aten::LeakyRelu to HLO
// LeakyRelu(x) = max(0, x) + negative_slop * min(0, x)
template <>
LogicalResult ConvertAtenOp<AtenLeakyReluOp>::matchAndRewrite(
    AtenLeakyReluOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  Location loc = op.getLoc();
  Value input = adaptor.self();
  Value negativeSlope = op.negative_slope();
  auto inputTy = input.getType().template dyn_cast<RankedTensorType>();
  if (!inputTy) {
    return op.emitError(
        "Only RankedTensorType is supported in Aten LeakyReLU op.");
  }

  double scaleValue;
  if (!matchPattern(negativeSlope, m_TorchConstantFloat(&scaleValue)))
    return op->emitError(
        "Currently only scalar constants are supported for "
        "negative_slope in MHLO operation");

  Value zeroVal = chlo::getConstantLike(rewriter, loc, 0.0, input);
  Value scaleVal = chlo::getConstantLike(rewriter, loc, scaleValue, input);

  Value leakyActivationVal = rewriter.create<mhlo::MulOp>(
      loc, getTypeConverter()->convertType(op.getType()), input, scaleVal);

  Value compareGtZero = rewriter.create<mhlo::CompareOp>(
      loc, input, zeroVal, mhlo::ComparisonDirection::GT);

  rewriter.replaceOpWithNewOp<mhlo::SelectOp>(
      op, inputTy, compareGtZero, input, leakyActivationVal);
  return success();
}

// Convert a Aten::Sigmoid to HLO
// Sigmoid(x) = 1.0 / (1.0 + exp(x))
template <>
LogicalResult ConvertAtenOp<AtenSigmoidOp>::matchAndRewrite(
    AtenSigmoidOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  Location loc = op.getLoc();
  Value input = adaptor.self();
  auto inputTy = input.getType().template dyn_cast<RankedTensorType>();
  if (!inputTy) {
    return op.emitError(
        "Only RankedTensorType is supported in Aten Sigmoid op.");
  }
  Value one = chlo::getConstantLike(rewriter, loc, 1.0, input);
  Value negVal = rewriter.create<mhlo::NegOp>(loc, input);
  Value expVal = rewriter.create<mhlo::ExpOp>(loc, negVal);
  Value addVal = rewriter.create<mhlo::AddOp>(loc, expVal, one);
  rewriter.replaceOpWithNewOp<mhlo::DivOp>(op, one, addVal);
  return success();
}

// Convert a Aten::SiLu to HLO
// SiLu(x) = x * Sigmoid(x)
template <>
LogicalResult ConvertAtenOp<AtenSiluOp>::matchAndRewrite(
    AtenSiluOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  Location loc = op.getLoc();
  Value input = adaptor.self();
  auto inputTy = input.getType().template dyn_cast<RankedTensorType>();
  if (!inputTy) {
    return op.emitError("Only RankedTensorType is supported in Aten SiLu op.");
  }
  Value sigmoid = rewriter.create<AtenSigmoidOp>(loc, inputTy, input);
  rewriter.replaceOpWithNewOp<mhlo::MulOp>(op, input, sigmoid);
  return success();
}

// Convert a Aten::GELU to HLO
// Gelu(x) = x * 1/2 * [1 + erf(x/(sqrt(2)))]
template <>
LogicalResult ConvertAtenOp<AtenGeluOp>::matchAndRewrite(
    AtenGeluOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  Location loc = op.getLoc();
  Value input = adaptor.self();
  auto inputTy = input.getType().template dyn_cast<RankedTensorType>();
  if (!inputTy) {
    return op.emitError("Only RankedTensorType is supported in Aten GELU op.");
  }

  auto elem_type = inputTy.getElementType();
  Value one = chlo::getConstantLike(rewriter, loc, 1.0, input);
  Value two = chlo::getConstantLike(rewriter, loc, 2.0, input);
  Value half = chlo::getConstantLike(rewriter, loc, 0.5, input);
  auto rsqrt_two = rewriter.create<mlir::mhlo::RsqrtOp>(loc, two);
  auto erf_element = rewriter.create<mhlo::MulOp>(loc, input, rsqrt_two);
  auto erf = rewriter.create<mlir::chlo::ErfOp>(loc, erf_element);
  auto erf_add = rewriter.create<mhlo::AddOp>(loc, erf, one);
  auto half_mul = rewriter.create<mhlo::MulOp>(loc, erf_add, half);
  rewriter.replaceOpWithNewOp<mhlo::MulOp>(op, input, half_mul);
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenGeluBackwardOp>::matchAndRewrite(
    AtenGeluBackwardOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  Location loc = op.getLoc();
  Value input = adaptor.self();
  Value grad_output = adaptor.self();
  Value half = chlo::getConstantLike(rewriter, loc, 0.5, input);
  Value one = chlo::getConstantLike(rewriter, loc, 1.0, input);
  Value m_2_sqrtpi = chlo::getConstantLike(
      rewriter, loc, 1.12837916709551257389615890312154517, input);
  Value m_sqrt1_2 = chlo::getConstantLike(
      rewriter, loc, 0.707106781186547524400844362104849039, input);

  Value alpha = rewriter.create<mhlo::MulOp>(loc, m_2_sqrtpi, m_sqrt1_2);
  alpha = rewriter.create<mhlo::MulOp>(loc, alpha, half);
  Value scratch = rewriter.create<chlo::ErfOp>(
      loc, rewriter.create<mhlo::MulOp>(loc, input, m_sqrt1_2));

  // dinput = exp(input * input * neg(half))
  Value dinput = rewriter.create<mhlo::MulOp>(loc, input, input);
  Value neg_half = rewriter.create<mhlo::NegOp>(loc, half);
  dinput = rewriter.create<mhlo::MulOp>(loc, input, neg_half);

  // result = grad_output * (half * (one + scratch) + input * dinput * alpha)
  Value half_one = rewriter.create<mhlo::AddOp>(loc, one, scratch);
  half_one = rewriter.create<mhlo::MulOp>(loc, half, half_one);

  Value input_alpha = rewriter.create<mhlo::MulOp>(loc, input, dinput);
  input_alpha = rewriter.create<mhlo::MulOp>(loc, input_alpha, alpha);

  Value result = rewriter.create<mhlo::AddOp>(loc, half_one, input_alpha);
  result = rewriter.create<mhlo::MulOp>(loc, grad_output, result);
  rewriter.replaceOp(op, {result});
  return success();
}

// Convert AtenEmptyMemoryFormatOp to const ones Tensor
// please note: The AtenEmptyMemoryFormatOp is not according to value sematic,
// and AtenEmptyMemoryFormatOp is a side-effected, to elimiate this op after
// Canonicalizer pass, we lower this op to a constant ones op, this is a tracy
// way.
template <>
LogicalResult ConvertAtenOp<AtenEmptyMemoryFormatOp>::matchAndRewrite(
    AtenEmptyMemoryFormatOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto outType = getTypeConverter()
                     ->convertType(op.getType())
                     .template dyn_cast<TensorType>();
  auto loc = op.getLoc();
  auto shape = adaptor.size();
  SmallVector<Value, 4> dimSizes;
  getListConstructElements(shape, dimSizes);
  // BladeDISC use i32 as shape
  std::for_each(dimSizes.begin(), dimSizes.end(), [&](Value& dSize) {
    dSize = rewriter.create<ToI64Op>(loc, dSize).getResult();
    // dimSize: cast i64 -> i32
    dSize = rewriter.create<arith::TruncIOp>(loc, rewriter.getI32Type(), dSize);
    return dSize;
  });

  auto mhloShape = rewriter.create<mlir::tensor::FromElementsOp>(loc, dimSizes);
  auto constOp =
      mhlo::getConstTensor<int32_t>(rewriter, op, {1.0}, {}).getValue();

  auto result = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
      loc, outType, constOp, mhloShape, rewriter.getI64TensorAttr({}));

  rewriter.replaceOp(op, {result});
  return success();
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
template <typename AtenOpT, typename MhloOpT>
class ConvertAtenReductionOp : public OpConversionPattern<AtenOpT> {
 public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;

  // Each variant must implement corresponding parameter parsing options
  virtual LogicalResult readReduceDimsAndKeepDims(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter,
      SmallVector<int64_t, 4>& reduceDims,
      DenseIntElementsAttr& reduceDimsAttr,
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
    auto selfTy = self.getType().dyn_cast<TensorType>();

    if (!selfTy)
      return op.emitError("Only tensor types supported in MHLO");

    auto outputTy = OpConversionPattern<AtenOpT>::getTypeConverter()
                        ->convertType(op.getType())
                        .template dyn_cast<RankedTensorType>();
    if (!outputTy)
      return op.emitError(
          "Only ranked tensor type outputs permitted for reduce_mean");
    SmallVector<int64_t, 4> reduceDims;
    DenseIntElementsAttr reduceDimsAttr;
    bool keepDims;
    if (failed(readReduceDimsAndKeepDims(
            op, adaptor, rewriter, reduceDims, reduceDimsAttr, keepDims)))
      return failure();

    auto loc = op.getLoc();
    auto elem_type = selfTy.getElementType();
    Type type = RankedTensorType::get(/*shape=*/{}, elem_type);
    DenseElementsAttr const_attr;
    if (auto float_ty = elem_type.dyn_cast_or_null<mlir::FloatType>()) {
      mlir::FloatAttr attr = mlir::FloatAttr::get(float_ty, 0);
      const_attr = mlir::DenseElementsAttr::get(type, attr);
    } else if (auto int_ty = elem_type.dyn_cast_or_null<mlir::IntegerType>()) {
      mlir::IntegerAttr attr = mlir::IntegerAttr::get(int_ty, 0);
      const_attr = mlir::DenseElementsAttr::get(type, attr);
    } else {
      return failure();
    }

    auto init_value =
        rewriter.create<mhlo::ConstantOp>(loc, type, const_attr).getResult();

    auto reduction =
        rewriter.create<mhlo::ReduceOp>(loc, self, init_value, reduceDimsAttr);

    {
      // Build body
      OpBuilder::InsertionGuard guard(rewriter);
      Block* block = rewriter.createBlock(&reduction.body());

      // Block arguments are scalars of the given element type.
      block->addArguments(type, loc);
      block->addArguments(type, loc);

      auto reduced = rewriter
                         .create<MhloOpT>(
                             loc, block->getArgument(0), block->getArgument(1))
                         .getResult();
      rewriter.create<mhlo::ReturnOp>(loc, reduced);
    }
    // post proccess for mean & keepDims
    auto result = reduction.getResult(0);
    bool isMean = isa<AtenMeanDimOp>(op);
    if (isMean || keepDims) {
      if (isMean) {
        auto numel = mhlo::getNumelOfTensor(rewriter, op, self);
        numel = scalarToMhloTensor(
            rewriter, op, numel, outputTy.getElementType(), {});
        result = rewriter.create<chlo::BroadcastDivOp>(
            loc, outputTy, result, numel, nullptr);
      }

      if (keepDims) {
        // unsqueeze the reduced dimensions
        auto dimSizes = mhlo::getDimSizesOfTensor(rewriter, op, self);
        for (auto d : reduceDims) {
          dimSizes[d] = rewriter.create<arith::ConstantOp>(
              loc, rewriter.getI32IntegerAttr(1));
        }
        // create new shape
        mlir::Value newShape =
            rewriter.create<tensor::FromElementsOp>(loc, dimSizes);
        // reshape the result
        result = rewriter.create<mhlo::DynamicReshapeOp>(
            loc, outputTy, result, newShape);
      }
    }

    if (!result)
      return failure();

    rewriter.replaceOpWithNewOp<mhlo::ConvertOp>(op, outputTy, result);

    return success();
  }
};

// This reduction op legalization template handles op variants that have
// explicit reduce_dims dimensions (provided as a list) and keep_dims
// parameters.
template <typename AtenOpT, typename MhloOpT>
class ConvertAtenMultipleDimsReductionOp
    : public ConvertAtenReductionOp<AtenOpT, MhloOpT> {
 public:
  using ConvertAtenReductionOp<AtenOpT, MhloOpT>::ConvertAtenReductionOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult readReduceDimsAndKeepDims(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter,
      SmallVector<int64_t, 4>& reduceDims,
      DenseIntElementsAttr& reduceDimsAttr,
      bool& keepDims) const override {
    if (!matchPattern(op.dim(), m_TorchConstantIntList(reduceDims)))
      return rewriter.notifyMatchFailure(
          op, "non-const dim parameter unsupported");

    auto self = adaptor.self();
    auto selfTy = self.getType().template cast<RankedTensorType>();
    auto rank = selfTy.getRank();
    std::transform(
        reduceDims.begin(),
        reduceDims.end(),
        reduceDims.begin(),
        [rank](int64_t d) -> int64_t { return (d + rank) % rank; });
    std::sort(reduceDims.begin(), reduceDims.end());
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
template <typename AtenOpT, typename MhloOpT>
class ConvertAtenOneDimReductionOp
    : public ConvertAtenReductionOp<AtenOpT, MhloOpT> {
 public:
  using ConvertAtenReductionOp<AtenOpT, MhloOpT>::ConvertAtenReductionOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult readReduceDimsAndKeepDims(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter,
      SmallVector<int64_t, 4>& reduceDims,
      DenseIntElementsAttr& reduceDimsAttr,
      bool& keepDims) const override {
    int64_t reduceDim;
    if (!matchPattern(op.dim(), m_TorchConstantInt(&reduceDim)))
      return rewriter.notifyMatchFailure(
          op, "non-const dim parameter unsupported");
    auto self = adaptor.self();
    auto selfTy = self.getType().template cast<RankedTensorType>();
    auto rank = selfTy.getRank();
    reduceDim = (reduceDim + rank) % rank;
    reduceDims.push_back(reduceDim);
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
template <typename AtenOpT, typename MhloOpT>
class ConvertAtenAllDimsReductionOp
    : public ConvertAtenReductionOp<AtenOpT, MhloOpT> {
 public:
  using ConvertAtenReductionOp<AtenOpT, MhloOpT>::ConvertAtenReductionOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult readReduceDimsAndKeepDims(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter,
      SmallVector<int64_t, 4>& reduceDims,
      DenseIntElementsAttr& reduceDimsAttr,
      bool& keepDims) const override {
    auto self = adaptor.self();
    auto selfTy = self.getType().template cast<RankedTensorType>();

    // Select all dims to reduce
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

// Perform the basic n-dim matmul operation encompassing the handling of
// broadcasting and dynamic shape propagation.
// All PyTorch ops that leverage matrix multiplication will derive this and
// implement their specialized input processing (e.g transpose), and output
// processing, e.g. GEMM or fully connected bias handling.
template <typename AtenOpT>
class ConvertAtenMatmulBaseOp : public OpConversionPattern<AtenOpT> {
 public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  // Each variant must implement corresponding parameter parsing options.
  // Maintain separate input read functions for each variant because it is not
  // necessarily true with all variants that the first two operands are the lhs
  // and rhs.
  virtual LogicalResult readMatMulInputs(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter,
      Value& lhs,
      Value& rhs) const {
    return rewriter.notifyMatchFailure(
        op,
        "Unimplemented matrix multiplication variant input parsing function");
  }
  LogicalResult performMatmul(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter,
      Value& lhs,
      Value& rhs,
      Value& output) const {
    auto lhsTy = lhs.getType().cast<RankedTensorType>();
    auto rhsTy = rhs.getType().cast<RankedTensorType>();

    auto lhsRank = lhsTy.getRank();
    auto rhsRank = rhsTy.getRank();

    auto lhsShape = lhsTy.getShape();
    auto rhsShape = rhsTy.getShape();

    auto lhsElemTy = lhsTy.getElementType();
    auto rhsElemTy = rhsTy.getElementType();

    if (lhsElemTy != rhsElemTy)
      return op.emitError("Matmul: input datatypes mismatched");
    if (lhsRank < 1 || rhsRank < 1) {
      return op.emitError("Matmul: inputs can't be 0-rank");
    }

    llvm::Optional<Value> product = llvm::None;
    if (rhsRank == 1) {
      if (lhsRank == 1) {
        // If both tensors are 1-dimensional, the dot product (scalar) is
        // returned.
        auto unsqzLhs = mhlo::getUnsqueezedTensor(rewriter, op, lhs, {0});
        product = mhlo::getMvDotProduct(rewriter, op, *unsqzLhs, rhs);
        product = mhlo::getZeroRankTensor(rewriter, op, *product);
      } else {
        // If the first argument is 2-dimensional and the second argument is
        // 1-dimensional, the matrix-vector product is returned.
        // NB: if lhsRank > 2 reshape it to rank 2.
        product = mhlo::getMvDotProduct(rewriter, op, lhs, rhs);
      }
    } else if (rhsRank == 2) {
      if (lhsRank == 1) {
        // If the first argument is 1-dimensional, a 1 is prepended to its
        // dimension for the purpose of the batched matrix multiply and removed
        // after.
        auto unsqzLhs = mhlo::getUnsqueezedTensor(rewriter, op, lhs, {0});
        product = mhlo::getMmDotProduct(rewriter, op, *unsqzLhs, rhs);
        std::vector<Value> collapDimSizes;
        std::tie(product, collapDimSizes) =
            mhlo::getCollapsedTensor(rewriter, op, *product, {-2, -1});
      } else {
        // If both arguments are 2-dimensional, the matrix-matrix product is
        // returned. NB: if lhsRank > 2 reshape it to rank 2.
        product = mhlo::getMmDotProduct(rewriter, op, lhs, rhs);
      }
    } else {
      // rhsRank > 2
      if (lhsRank == 1) {
        // If the first argument is 1-dimensional, a 1 is prepended to its
        // dimension for the purpose of the batched matrix multiply and removed
        // after.
        auto unsqzLhs = mhlo::getUnsqueezedTensor(rewriter, op, lhs, {0});
        product = mhlo::getBmmDotProduct(rewriter, op, *unsqzLhs, rhs);
        std::vector<Value> collapDimSizes;
        std::tie(product, collapDimSizes) =
            mhlo::getCollapsedTensor(rewriter, op, *product, {-2, -1});
      } else {
        product = mhlo::getBmmDotProduct(rewriter, op, lhs, rhs);
      }
    }
    if (product) {
      output = *product;
    } else {
      return op.emitError("Matmul: conversion failed");
    }
    return success();
  }

  // The default version just reads two inputs, computes output and returns it.
  // Other versions may add a bias, apply GEMM-style alpha/beta scaling etc.
  virtual LogicalResult matchAndRewrite(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value lhs, rhs;
    if (failed(readMatMulInputs(op, adaptor, rewriter, lhs, rhs)))
      return op.emitError("Failed to read matmul inputs");

    Value output;

    if (failed(performMatmul(op, adaptor, rewriter, lhs, rhs, output)))
      return op.emitError("Failed to perform matmul operation");

    rewriter.replaceOpWithNewOp<mhlo::ConvertOp>(
        op,
        OpConversionPattern<AtenOpT>::getTypeConverter()
            ->convertType(op.getType())
            .template cast<RankedTensorType>(),
        output);

    return success();
  }
};

// Legalizes the torch.matmul op for general n-dim matmul.
template <typename AtenOpT>
class ConvertAtenMatMulOp : public ConvertAtenMatmulBaseOp<AtenOpT> {
 public:
  using ConvertAtenMatmulBaseOp<AtenOpT>::ConvertAtenMatmulBaseOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult readMatMulInputs(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter,
      Value& lhs,
      Value& rhs) const override {
    lhs = adaptor.self();
    auto lhsTy = lhs.getType().cast<RankedTensorType>();

    rhs = adaptor.other();
    auto rhsTy = rhs.getType().cast<RankedTensorType>();

    if (!lhsTy || !rhsTy)
      return op.emitError("Only ranked tensor types supported in MHLO matmul");

    return success();
  }
};

// Implements handling of aten.mm and aten.bmm ops.
template <typename AtenOpT>
class ConvertAtenMmOp : public ConvertAtenMatmulBaseOp<AtenOpT> {
 public:
  using ConvertAtenMatmulBaseOp<AtenOpT>::ConvertAtenMatmulBaseOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult readMatMulInputs(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter,
      Value& lhs,
      Value& rhs) const override {
    lhs = adaptor.self();
    auto lhsTy = lhs.getType().cast<RankedTensorType>();

    rhs = adaptor.mat2();
    auto rhsTy = rhs.getType().cast<RankedTensorType>();

    if (!lhsTy || !rhsTy)
      return op.emitError("Only ranked tensor types supported in MHLO matmul");

    auto lhsRank = lhsTy.getRank();
    auto rhsRank = rhsTy.getRank();

    if (isa<AtenMmOp>(op)) {
      // Mm takes two 2D tensors.
      if (lhsRank != 2 || rhsRank != 2)
        return op.emitError("aten.mm called but matrix rank != 2");
    } else if (isa<AtenBmmOp>(op)) {
      // Bmm takes two 3D tensors.
      if (lhsRank != 3 || rhsRank != 3)
        return op.emitError("aten.bmm called but matrix rank != 3");
    }

    return success();
  }
};

// Implements handling of aten.linear op.
template <typename AtenOpT>
class ConvertAtenLinearOp : public ConvertAtenMatmulBaseOp<AtenOpT> {
 public:
  using ConvertAtenMatmulBaseOp<AtenOpT>::ConvertAtenMatmulBaseOp;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult readMatMulInputs(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter,
      Value& lhs,
      Value& rhs) const override {
    lhs = adaptor.input();
    auto lhsTy = lhs.getType().cast<RankedTensorType>();

    rhs = adaptor.weight();
    auto rhsTy = rhs.getType().cast<RankedTensorType>();

    if (!lhsTy || !rhsTy)
      return op.emitError("Only ranked tensor types supported in MHLO matmul");

    auto lhsRank = lhsTy.getRank();
    auto rhsRank = rhsTy.getRank();

    if (lhsRank != 2 && lhsRank != 3)
      return op.emitError("aten.Linear called but input rank not 2 or 3");
    if (rhsRank != 2 && rhsRank != 3)
      return op.emitError("aten.Linear called but weight rank not 2 or 3");

    return success();
  }
  // Override the default rewriter to perform RHS transpose and bias addition
  // as well.
  LogicalResult matchAndRewrite(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value lhs, rhs;

    if (failed(readMatMulInputs(op, adaptor, rewriter, lhs, rhs)))
      return op.emitError("Failed to read matmul inputs");

    // The aten.Linear op has a bias tensor that is added to the matmul
    // output.
    auto bias = adaptor.bias();
    auto biasTy = bias.getType();

    // MHLO does not mandate that elementwise op tensors need to be ranked.
    if (!biasTy.template isa<Torch::NoneType>() &&
        !biasTy.template isa<TensorType>())
      return op.emitError(
          "Only tensor types supported in GEMM to "
          "MHLO for bias tensor");

    // weight.T
    auto weightT = mhlo::getPermutedTensor(rewriter, op, rhs, {1, 0});
    auto product = mhlo::getMmDotProduct(rewriter, op, lhs, weightT);
    Value matmulOutput;
    if (product) {
      matmulOutput = *product;
    } else {
      return op.emitError("Failed to perform matmul operation");
    }

    Value matmulPlusBias = matmulOutput;

    if (!biasTy.template isa<Torch::NoneType>()) {
      // Bias addition broadcasts to the matmul output shape.
      matmulPlusBias = rewriter
                           .create<chlo::BroadcastAddOp>(
                               op->getLoc(),
                               matmulOutput.getType(),
                               matmulOutput,
                               bias,
                               nullptr)
                           .getResult();
    }

    rewriter.replaceOpWithNewOp<mhlo::ConvertOp>(
        op,
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()),
        matmulPlusBias);
    return success();
  }
};

template <>
LogicalResult ConvertAtenOp<AtenIndexSelectOp>::matchAndRewrite(
    AtenIndexSelectOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto self = adaptor.self();
  auto selfTy = self.getType().template cast<RankedTensorType>();
  if (!selfTy)
    return op.emitError("Only ranked tensor types supported in MHLO");
  int64_t dim;
  if (!matchPattern(op.dim(), m_TorchConstantInt(&dim)))
    return rewriter.notifyMatchFailure(
        op, "Only constant dim is currently supported");

  Value sliced =
      mhlo::getGatheredTensor(rewriter, op, self, adaptor.index(), dim);

  rewriter.replaceOpWithNewOp<mhlo::ConvertOp>(
      op, getTypeConverter()->convertType(op.getType()), sliced);

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenSqueezeDimOp>::matchAndRewrite(
    AtenSqueezeDimOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto self = adaptor.self();
  auto selfTy = self.getType().template cast<RankedTensorType>();
  if (!selfTy)
    return op.emitError("Only ranked tensor types supported in MHLO");
  int64_t dim;
  if (!matchPattern(op.dim(), m_TorchConstantInt(&dim)))
    return rewriter.notifyMatchFailure(
        op, "Only constant dim is currently supported");

  auto rank = selfTy.getRank();
  if (rank == 0) {
    return rewriter.notifyMatchFailure(
        op, "The rank of tensor must be greater than 0");
  }

  dim = (dim + rank) % rank;
  if (selfTy.getShape()[dim] != 1) {
    if (selfTy.getShape()[dim] == ShapedType::kDynamicSize) {
      return rewriter.notifyMatchFailure(
          op, "The size of the dimension being squeezed is can't be unknown");
    } else {
      rewriter.replaceOpWithNewOp<mhlo::ConvertOp>(
          op, getTypeConverter()->convertType(op.getType()), self);
      return success();
    }
  }

  auto dims = mhlo::rangeIndices(0, rank);
  dims.erase(dims.begin() + dim);
  auto newDimSizes = mhlo::getDimSizesOfTensor(rewriter, op, self, dims);
  auto mhloShape =
      rewriter.create<tensor::FromElementsOp>(op.getLoc(), newDimSizes);
  rewriter.replaceOpWithNewOp<mhlo::DynamicReshapeOp>(
      op, getTypeConverter()->convertType(op.getType()), self, mhloShape);
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenSqueezeOp>::matchAndRewrite(
    AtenSqueezeOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto self = adaptor.self();
  auto selfTy = self.getType().template cast<RankedTensorType>();
  if (!selfTy)
    return op.emitError("Only ranked tensor types supported in MHLO");

  auto rank = selfTy.getRank();
  if (rank == 0) {
    return rewriter.notifyMatchFailure(
        op, "The rank of tensor must be greater than 0");
  }

  SmallVector<int64_t, 4> dims;
  dims.reserve(rank);
  for (int r = 0; r < rank; ++r) {
    auto dSize = selfTy.getShape()[r];
    if (dSize == ShapedType::kDynamicSize) {
      return rewriter.notifyMatchFailure(
          op, "The size of the dimension being squeezed is can't be unknown");
    }
    if (dSize != 1) {
      dims.push_back(r);
    }
  }

  auto newDimSizes = mhlo::getDimSizesOfTensor(rewriter, op, self, dims);
  auto mhloShape =
      rewriter.create<tensor::FromElementsOp>(op.getLoc(), newDimSizes);
  rewriter.replaceOpWithNewOp<mhlo::DynamicReshapeOp>(
      op, getTypeConverter()->convertType(op.getType()), self, mhloShape);
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenSliceTensorOp>::matchAndRewrite(
    AtenSliceTensorOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto self = adaptor.self();
  auto selfTy = self.getType().template cast<RankedTensorType>();
  if (!selfTy)
    return op.emitError("Only ranked tensor types supported in MHLO Rsub");
  int64_t dim;
  if (!matchPattern(op.dim(), m_TorchConstantInt(&dim)))
    return rewriter.notifyMatchFailure(
        op, "Only constant dim is currently supported");

  auto getOptionalVal = [&](Value val) -> llvm::Optional<Value> {
    if (val.getType().isa<Torch::NoneType>()) {
      return llvm::None;
    } else {
      return val;
    }
  };

  llvm::Optional<Value> start = getOptionalVal(adaptor.start());
  llvm::Optional<Value> end = getOptionalVal(adaptor.end());
  llvm::Optional<Value> step = getOptionalVal(adaptor.step());

  Value sliced =
      mhlo::getDynamicSlice(rewriter, op, self, start, end, step, dim);
  rewriter.replaceOpWithNewOp<mhlo::ConvertOp>(
      op, getTypeConverter()->convertType(op.getType()), sliced);

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

  auto multTensor = rewriter.create<chlo::BroadcastMulOp>(
      op->getLoc(),
      getTypeConverter()->convertType(op.getType()),
      self,
      alphaTensor,
      nullptr);

  rewriter.replaceOpWithNewOp<chlo::BroadcastSubOp>(
      op,
      getTypeConverter()->convertType(op.getType()),
      otherTensor,
      multTensor,
      nullptr);

  return success();
}

// Torch constants are converted to mhlo.const .
template <>
LogicalResult ConvertAtenOp<ValueTensorLiteralOp>::matchAndRewrite(
    ValueTensorLiteralOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  MLIRContext* context = op->getContext();
  if (auto elements = op.valueAttr().dyn_cast<DenseIntElementsAttr>()) {
    Type elemTy = op.valueAttr().getElementType();
    unsigned bitWidth = elemTy.getIntOrFloatBitWidth();
    if (elemTy.isUnsignedInteger()) {
      Type builtinTensorElemTy = IntegerType::get(
          context, bitWidth, IntegerType::SignednessSemantics::Unsigned);
      rewriter.replaceOpWithNewOp<mhlo::ConstantOp>(
          op, elements.mapValues(builtinTensorElemTy, [&](const APInt& v) {
            return APInt(bitWidth, v.getZExtValue());
          }));
    } else {
      Type builtinTensorElemTy = IntegerType::get(context, bitWidth);
      rewriter.replaceOpWithNewOp<mhlo::ConstantOp>(
          op, elements.mapValues(builtinTensorElemTy, [&](const APInt& v) {
            return APInt(bitWidth, v.getSExtValue());
          }));
    }
    return success();
  }
  rewriter.replaceOpWithNewOp<mhlo::ConstantOp>(op, op.valueAttr());
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenPermuteOp>::matchAndRewrite(
    AtenPermuteOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  // Not a ranked tensor type
  auto selfType = adaptor.self().getType().dyn_cast<RankedTensorType>();
  if (!selfType)
    return op.emitError(
        "Only ranked tensor types with static shapes are currently supported");

  SmallVector<int64_t> dimListInt;
  if (!matchPattern(adaptor.dims(), m_TorchConstantIntList(dimListInt)))
    return rewriter.notifyMatchFailure(
        op, "Only constant dimensions are currently supported");

  Value transposed =
      mhlo::getPermutedTensor(rewriter, op, adaptor.self(), dimListInt);

  rewriter.replaceOpWithNewOp<mhlo::ConvertOp>(
      op, getTypeConverter()->convertType(op.getType()), transposed);

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenTOp>::matchAndRewrite(
    AtenTOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  // Not a ranked tensor type
  auto selfType = adaptor.self().getType().dyn_cast<RankedTensorType>();
  if (!selfType)
    return op.emitError("Only ranked tensor types are currently supported");

  if (selfType.getRank() != 2)
    return rewriter.notifyMatchFailure(op, "Only 2-rank tensors are allowed");

  Value transposed =
      mhlo::getPermutedTensor(rewriter, op, adaptor.self(), {1, 0});

  rewriter.replaceOpWithNewOp<mhlo::ConvertOp>(
      op, getTypeConverter()->convertType(op.getType()), transposed);

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenTransposeIntOp>::matchAndRewrite(
    AtenTransposeIntOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  // Not a ranked tensor type
  auto selfType = adaptor.self().getType().dyn_cast<RankedTensorType>();
  if (!selfType)
    return op.emitError("Only ranked tensor types are currently supported");

  if (selfType.getRank() < 2)
    return rewriter.notifyMatchFailure(op, "Expect tensor rank greater than 1");

  int64_t dim0, dim1;
  if (!matchPattern(op.dim0(), m_TorchConstantInt(&dim0)))
    return rewriter.notifyMatchFailure(
        op, "Only constant dim0 are currently supported");
  if (!matchPattern(op.dim1(), m_TorchConstantInt(&dim1)))
    return rewriter.notifyMatchFailure(
        op, "Only constant dim1 are currently supported");

  auto rank = selfType.getRank();
  dim0 = (dim0 + rank) % rank;
  dim1 = (dim1 + rank) % rank;
  auto permutations = mhlo::rangeIndices(0, rank);
  std::swap(permutations[dim0], permutations[dim1]);

  Value transposed =
      mhlo::getPermutedTensor(rewriter, op, adaptor.self(), permutations);
  rewriter.replaceOpWithNewOp<mhlo::ConvertOp>(
      op, getTypeConverter()->convertType(op.getType()), transposed);

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
LogicalResult ConvertAtenOp<AtenUnsqueezeOp>::matchAndRewrite(
    AtenUnsqueezeOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  // Not a tensor type.
  auto selfType = adaptor.self().getType().dyn_cast<TensorType>();
  if (!selfType) {
    return op.emitError("Only tensor types are currently supported");
  }

  auto selfRank = selfType.getRank();
  auto selfElemTy = selfType.getElementType();

  int64_t dim;
  if (!matchPattern(op.dim(), m_TorchConstantInt(&dim)))
    return op->emitError("dim must be a Scalar constant");

  auto unsqzTensor =
      mhlo::getUnsqueezedTensor(rewriter, op, adaptor.self(), {dim});
  rewriter.replaceOp(op, *unsqzTensor);
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
LogicalResult ConvertAtenOp<AtenNumelOp>::matchAndRewrite(
    AtenNumelOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto outType = op.getType().dyn_cast<TensorType>();

  Value numel = mhlo::getNumelOfTensor(rewriter, op, adaptor.self());
  rewriter.replaceOpWithNewOp<arith::ExtSIOp>(
      op, getTypeConverter()->convertType(op.getType()), numel);
  return success();
}

template <>
LogicalResult ConvertAtenOp<PrimNumToTensorScalarOp>::matchAndRewrite(
    PrimNumToTensorScalarOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto outType =
      OpConversionPattern<PrimNumToTensorScalarOp>::getTypeConverter()
          ->convertType(op.getType())
          .template dyn_cast<TensorType>();
  if (!outType)
    return op.emitError("output should be TensorType");

  Value constOp;
  double val;
  if (!matchPattern(op.a(), m_TorchConstantFloat(&val))) {
    constOp = scalarToMhloTensor(
        rewriter, op, adaptor.a(), outType.getElementType(), {});
    rewriter.replaceOp(op, constOp);
    return success();
  }
  // return op.emitError("input should be constant");

  if (failed(torchScalarToMhloTensor(
          rewriter,
          op,
          op.a(),
          constOp,
          outType.getElementType(),
          outType.getShape())))
    return op.emitError("Supplied value must be a Scalar constant");
  rewriter.replaceOp(op, constOp);
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenTensorIntOp>::matchAndRewrite(
    AtenTensorIntOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(
      op,
      getTypeConverter()->convertType(op.getType()),
      ArrayRef<Value>{adaptor.t()});
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
LogicalResult ConvertAtenOp<TensorStaticInfoCastOp>::matchAndRewrite(
    TensorStaticInfoCastOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  // Not a tensor type.
  auto operandType = adaptor.operand().getType().dyn_cast<TensorType>();
  if (!operandType)
    return op.emitError("Only tensor types are currently supported");

  rewriter.replaceOpWithNewOp<mhlo::ConvertOp>(
      op, getTypeConverter()->convertType(op.getType()), adaptor.operand());

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenFlipOp>::matchAndRewrite(
    AtenFlipOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto self = adaptor.self();
  auto selfTy = self.getType().template cast<RankedTensorType>();
  if (!selfTy)
    return op.emitError("Only ranked tensor types supported in MHLO");

  SmallVector<int64_t, 4> dimListInt;
  if (!matchPattern(op.dims(), m_TorchConstantIntList(dimListInt)))
    return rewriter.notifyMatchFailure(
        op, "Only constant dims are currently supported");

  rewriter.replaceOpWithNewOp<mlir::mhlo::ReverseOp>(
      op,
      getTypeConverter()->convertType(op.getType()),
      self,
      BuildI64ElementsAttr(rewriter, dimListInt));

  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenRollOp>::matchAndRewrite(
    AtenRollOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto self = adaptor.self();
  auto selfTy = self.getType().template cast<RankedTensorType>();
  if (!selfTy)
    return op.emitError("Only ranked tensor types supported in MHLO");

  SmallVector<int64_t, 4> shiftListInt;
  if (!matchPattern(op.shifts(), m_TorchConstantIntList(shiftListInt)))
    return rewriter.notifyMatchFailure(
        op, "Only constant shifts are currently supported");

  SmallVector<int64_t, 4> dimListInt;
  if (!matchPattern(op.dims(), m_TorchConstantIntList(dimListInt)))
    return rewriter.notifyMatchFailure(
        op, "Only constant dims are currently supported");

  auto roll = self;
  for (size_t d = 0; d < dimListInt.size(); ++d) {
    roll =
        mhlo::getRollTensor(rewriter, op, roll, shiftListInt[d], dimListInt[d]);
  }
  rewriter.replaceOp(op, roll);
  return success();
}

// This defines a template to construct ops whose legalizations are
// specialized.
template <typename AtenOpT>
class ConvertAtenViewOp : public OpConversionPattern<AtenOpT> {
 public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;

  LogicalResult matchAndRewrite(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    // Not a tensor type.
    auto rankType =
        adaptor.self().getType().template dyn_cast<RankedTensorType>();
    if (!rankType)
      return op.emitError("Only ranked tensor types are currently supported");

    SmallVector<Value, 4> dimSizes;
    if (!getAtenViewOpSizes(op, adaptor, rewriter, dimSizes)) {
      return op.emitError("Dims size must be a list of Scalar");
    }

    auto loc = op.getLoc();
    auto newRank = dimSizes.size();
    if (newRank == 0) {
      rewriter.replaceOpWithNewOp<mhlo::ReshapeOp>(
          op,
          OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
              op.getType()),
          adaptor.self());
      return success();
    }

    std::for_each(dimSizes.begin(), dimSizes.end(), [&](Value& dSize) {
      dSize = rewriter.create<ToI64Op>(loc, dSize).getResult();
      // dimSize: cast i64 -> i32
      dSize =
          rewriter.create<arith::TruncIOp>(loc, rewriter.getI32Type(), dSize);
      return dSize;
    });

    auto mhloShape = rewriter.create<tensor::FromElementsOp>(loc, dimSizes);
    rewriter.replaceOpWithNewOp<chlo::DynamicReshapeOp>(
        op,
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()),
        adaptor.self(),
        mhloShape);
    return success();
  }

  bool getAtenViewOpSizes(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter,
      SmallVector<Value, 4>& dimSizes) const;
};

template <>
bool ConvertAtenViewOp<AtenViewOp>::getAtenViewOpSizes(
    AtenViewOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter,
    SmallVector<Value, 4>& dimSizes) const {
  return getListConstructElements(adaptor.size(), dimSizes);
}

template <>
bool ConvertAtenViewOp<AtenReshapeOp>::getAtenViewOpSizes(
    AtenReshapeOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter,
    SmallVector<Value, 4>& dimSizes) const {
  return getListConstructElements(adaptor.shape(), dimSizes);
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
      BuildI64ElementsAttr(rewriter, mhlo::rangeIndices(leadingRank, newRank));
  rewriter.replaceOpWithNewOp<mhlo::DynamicBroadcastInDimOp>(
      op,
      getTypeConverter()->convertType(op.getType()),
      adaptor.self(),
      mhloShape,
      broadcastDims);
  return success();
}

StringAttr PackRandomUniformBackendConfig(
    IntegerAttr seed,
    IntegerAttr seed2,
    PatternRewriter* rewriter) {
  mhlo_disc::RngUniformBackendConfig config(
      seed.getValue().getSExtValue(), seed2.getValue().getSExtValue());
  std::string str;
  llvm::raw_string_ostream ostream(str);
  ostream << ::llvm::json::toJSON(config);
  return rewriter->getStringAttr(ostream.str());
}

template <>
LogicalResult ConvertAtenOp<ValsemVariantAtenUniformOp>::matchAndRewrite(
    ValsemVariantAtenUniformOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto inputTy = adaptor.self().getType().template cast<RankedTensorType>();
  auto loc = op.getLoc();
  if (!inputTy) {
    op.emitError("input should be ranked tensor type.");
  }
  auto definingOp = op.self().getDefiningOp();
  auto shape = definingOp->getOperand(0);
  SmallVector<Value, 4> dimSizes;
  getListConstructElements(shape, dimSizes);
  std::for_each(dimSizes.begin(), dimSizes.end(), [&](Value& dSize) {
    dSize = rewriter.create<ToI64Op>(loc, dSize).getResult();
    // dimSize: cast i64 -> i32
    dSize = rewriter.create<arith::TruncIOp>(loc, rewriter.getI32Type(), dSize);
    return dSize;
  });

  auto mhloShape =
      rewriter.create<tensor::FromElementsOp>(op.getLoc(), dimSizes);

  double fromDoubleValue, toDoubleValue;
  if (!matchPattern(op.from(), m_TorchConstantFloat(&fromDoubleValue))) {
    op.emitError("operand #1 should be scalar");
  }
  if (!matchPattern(op.to(), m_TorchConstantFloat(&toDoubleValue))) {
    op.emitError("operand #2 should be scalar");
  }
  Value fromTensor = rewriter.create<mhlo::ConstOp>(
      op.getLoc(),
      rewriter.getFloatAttr(inputTy.getElementType(), fromDoubleValue));
  Value toTensor = rewriter.create<mhlo::ConstOp>(
      op.getLoc(),
      rewriter.getFloatAttr(inputTy.getElementType(), toDoubleValue));

  auto cfg = PackRandomUniformBackendConfig(
      rewriter.getIntegerAttr(rewriter.getI32Type(), 1),
      rewriter.getIntegerAttr(rewriter.getI32Type(), 2),
      &rewriter);
  auto outType = getTypeConverter()
                     ->convertType(op.getType())
                     .template dyn_cast<TensorType>();

  auto custom_call_op = rewriter.create<mhlo_disc::CustomCallOp>(
      op.getLoc(),
      TypeRange{outType},
      ValueRange{fromTensor, toTensor, mhloShape},
      rewriter.getStringAttr("rng_uniform"),
      rewriter.getBoolAttr(false),
      cfg);
  rewriter.replaceOp(op, {custom_call_op.getResult(0)});
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
    auto inputTy = input.getType().template dyn_cast<RankedTensorType>();
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
      if (!outType)
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
    registry.insert<mhlo_disc::MhloDiscDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<
        chlo::ChloDialect,
        mhlo::MhloDialect,
        mhlo_disc::MhloDiscDialect,
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
    INSERT_UNARY_PATTERN(AtenItemOp, tensor::ExtractOp)

    // It's tricky that ConvertOp will use type from the return,
    // but not from the operand here.
    INSERT_UNARY_PATTERN(AtenContiguousOp, mhlo::ConvertOp)
    INSERT_UNARY_PATTERN(AtenToDtypeOp, mhlo::ConvertOp);
    INSERT_UNARY_PATTERN(AtenTypeAsOp, mhlo::ConvertOp);
#undef INSERT_UNARY_PATTERN

#define INSERT_BINARY_BROADCAST_PATTERN(AtenOp, MhloOp)       \
  target.addIllegalOp<AtenOp>();                              \
  patterns.add<ConvertAtenBinaryBroadcastOp<AtenOp, MhloOp>>( \
      typeConverter, context);
    INSERT_BINARY_BROADCAST_PATTERN(AtenMaximumOp, chlo::BroadcastMaxOp)
    INSERT_BINARY_BROADCAST_PATTERN(AtenMinimumOp, chlo::BroadcastMinOp)
    INSERT_BINARY_BROADCAST_PATTERN(Aten__And__TensorOp, chlo::BroadcastAndOp)
    // INSERT_BINARY_BROADCAST_PATTERN(Aten__Or__TensorOp, chlo::BroadcastOrOp)
    // INSERT_BINARY_BROADCAST_PATTERN(Aten__XOr__TensorOp,
    // chlo::BroadcastXorOp)
#undef INSERT_BINARY_BROADCAST_PATTERN

#define INSERT_BINARY_COMPARE_PATTERN(AtenOp, Direction)       \
  target.addIllegalOp<AtenOp>();                               \
  patterns.add<ConvertAtenBinaryCompareOp<AtenOp, Direction>>( \
      typeConverter, context);
    INSERT_BINARY_COMPARE_PATTERN(
        AtenGtTensorOp, mhlo::ComparisonDirection::GT);
    INSERT_BINARY_COMPARE_PATTERN(
        AtenGtScalarOp, mhlo::ComparisonDirection::GT);
    // INSERT_BINARY_COMPARE_PATTERN(AtenGeTensorOp,
    // mhlo::ComparisonDirection::GE);
    INSERT_BINARY_COMPARE_PATTERN(
        AtenGeScalarOp, mhlo::ComparisonDirection::GE);
    INSERT_BINARY_COMPARE_PATTERN(
        AtenEqTensorOp, mhlo::ComparisonDirection::EQ);
    INSERT_BINARY_COMPARE_PATTERN(
        AtenEqScalarOp, mhlo::ComparisonDirection::EQ);
    INSERT_BINARY_COMPARE_PATTERN(
        AtenNeTensorOp, mhlo::ComparisonDirection::NE);
    INSERT_BINARY_COMPARE_PATTERN(
        AtenNeScalarOp, mhlo::ComparisonDirection::NE);
    // INSERT_BINARY_COMPARE_PATTERN(AtenLeTensorOp,
    // mhlo::ComparisonDirection::LE);
    INSERT_BINARY_COMPARE_PATTERN(
        AtenLeScalarOp, mhlo::ComparisonDirection::LE);
    INSERT_BINARY_COMPARE_PATTERN(
        AtenLtTensorOp, mhlo::ComparisonDirection::LT);
    INSERT_BINARY_COMPARE_PATTERN(
        AtenLtScalarOp, mhlo::ComparisonDirection::LT);
#undef INSERT_BINARY_COMPARE_PATTERN

#define INSERT_BINARY_ADDSUB_PATTERN(AtenOp, MhloOp) \
  target.addIllegalOp<AtenOp>();                     \
  patterns.add<ConvertAtenAddSubOp<AtenOp, MhloOp>>(typeConverter, context);
    // INSERT_BINARY_ADDSUB_PATTERN(AtenAddTensorOp, mhlo::AddOp)
    // INSERT_BINARY_ADDSUB_PATTERN(AtenAddScalarOp, mhlo::AddOp)
    INSERT_BINARY_ADDSUB_PATTERN(AtenAddTensorOp, chlo::BroadcastAddOp)
    INSERT_BINARY_ADDSUB_PATTERN(AtenAddScalarOp, chlo::BroadcastAddOp)
    // INSERT_BINARY_ADDSUB_PATTERN(AtenSubTensorOp, mhlo::SubtractOp)
    // INSERT_BINARY_ADDSUB_PATTERN(AtenSubScalarOp, mhlo::SubtractOp)
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
    INSERT_BINARY_DIV_PATTERN(AtenDivTensorOp)
    INSERT_BINARY_DIV_PATTERN(AtenDivTensorModeOp)
    INSERT_BINARY_DIV_PATTERN(AtenDivScalarOp)
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
    INSERT_FILL_SCALAR_PATTERN(ValsemVariantAtenFillScalarOp);
#undef INSERT_FILL_SCALAR_PATTERN

#define INSERT_MATMUL_ATENOP_PATTERN(AtenOp) \
  target.addIllegalOp<AtenOp>();             \
  patterns.add<ConvertAtenMatMulOp<AtenOp>>(typeConverter, context);
    INSERT_MATMUL_ATENOP_PATTERN(AtenMatmulOp);
#undef INSERT_MATMUL_ATEMOP_PATTERN

#define INSERT_MM_ATENOP_PATTERN(AtenOp) \
  target.addIllegalOp<AtenOp>();         \
  patterns.add<ConvertAtenMmOp<AtenOp>>(typeConverter, context);
    INSERT_MM_ATENOP_PATTERN(AtenMmOp);
    INSERT_MM_ATENOP_PATTERN(AtenBmmOp);
#undef INSERT_MM_ATEMOP_PATTERN

#define INSERT_LINEAR_ATENOP_PATTERN(AtenOp) \
  target.addIllegalOp<AtenOp>();             \
  patterns.add<ConvertAtenLinearOp<AtenOp>>(typeConverter, context);
    INSERT_LINEAR_ATENOP_PATTERN(AtenLinearOp);
#undef INSERT_LINEAR_ATEMOP_PATTERN

#define INSERT_ATENOP_PATTERN(AtenOp) \
  target.addIllegalOp<AtenOp>();      \
  patterns.add<ConvertAtenOp<AtenOp>>(typeConverter, context);
    INSERT_ATENOP_PATTERN(AtenBroadcastToOp);
    INSERT_ATENOP_PATTERN(AtenSigmoidOp);
    INSERT_ATENOP_PATTERN(AtenReluOp);
    INSERT_ATENOP_PATTERN(AtenRelu6Op);
    INSERT_ATENOP_PATTERN(AtenLeakyReluOp);
    INSERT_ATENOP_PATTERN(AtenSiluOp);
    INSERT_ATENOP_PATTERN(AtenGeluOp);
    INSERT_ATENOP_PATTERN(AtenSizeIntOp);
    INSERT_ATENOP_PATTERN(AtenTanhOp);
    // INSERT_ATENOP_PATTERN(AtenArgmaxOp);
    INSERT_ATENOP_PATTERN(AtenPowTensorScalarOp);
    INSERT_ATENOP_PATTERN(AtenRsubScalarOp);
    INSERT_ATENOP_PATTERN(ValueTensorLiteralOp);
    // INSERT_ATENOP_PATTERN(AtenReshapeOp);
    // INSERT_ATENOP_PATTERN(AtenFlattenUsingIntsOp);
    INSERT_ATENOP_PATTERN(AtenPermuteOp);
    INSERT_ATENOP_PATTERN(AtenTOp);
    INSERT_ATENOP_PATTERN(AtenTransposeIntOp);
    INSERT_ATENOP_PATTERN(AtenLog2Op);
    INSERT_ATENOP_PATTERN(AtenUnsqueezeOp);
    INSERT_ATENOP_PATTERN(AtenDropoutOp);
    INSERT_ATENOP_PATTERN(AtenNumelOp);
    INSERT_ATENOP_PATTERN(PrimNumToTensorScalarOp);
    INSERT_ATENOP_PATTERN(AtenTensorIntOp);
    INSERT_ATENOP_PATTERN(AtenSliceTensorOp);
    INSERT_ATENOP_PATTERN(AtenSqueezeOp);
    INSERT_ATENOP_PATTERN(AtenSqueezeDimOp);
    INSERT_ATENOP_PATTERN(AtenFlipOp);
    INSERT_ATENOP_PATTERN(AtenIndexSelectOp);
    INSERT_ATENOP_PATTERN(AtenRollOp);
    INSERT_ATENOP_PATTERN(ValsemVariantAtenUniformOp);
    INSERT_ATENOP_PATTERN(TensorStaticInfoCastOp);
    INSERT_ATENOP_PATTERN(AtenGeluBackwardOp);
    INSERT_ATENOP_PATTERN(AtenEmptyMemoryFormatOp);
#undef INSERT_ATENOP_PATTERN

#define INSERT_VIEW_OP_PATTERN(AtenOp) \
  target.addIllegalOp<AtenOp>();       \
  patterns.add<ConvertAtenViewOp<AtenOp>>(typeConverter, context);
    INSERT_VIEW_OP_PATTERN(AtenViewOp);
    INSERT_VIEW_OP_PATTERN(AtenReshapeOp);
#undef INSERT_VIEW_OP_PATTERN

#define INSERT_NDIMS_REDUCTION_OP_PATTERN(AtenOp, MhloOp)           \
  target.addIllegalOp<AtenOp>();                                    \
  patterns.add<ConvertAtenMultipleDimsReductionOp<AtenOp, MhloOp>>( \
      typeConverter, context);
    INSERT_NDIMS_REDUCTION_OP_PATTERN(AtenSumDimIntListOp, mhlo::AddOp)
#undef INSERT_NDIMS_REDUCTION_OP_PATTERN

#define INSERT_ONEDIM_REDUCTION_OP_PATTERN(AtenOp, MhloOp)    \
  target.addIllegalOp<AtenOp>();                              \
  patterns.add<ConvertAtenOneDimReductionOp<AtenOp, MhloOp>>( \
      typeConverter, context);
    INSERT_ONEDIM_REDUCTION_OP_PATTERN(AtenMeanDimOp, mhlo::AddOp)
    INSERT_ONEDIM_REDUCTION_OP_PATTERN(AtenAnyDimOp, mhlo::OrOp)
#undef INSERT_ONEDIM_REDUCTION_OP_PATTERN

#define INSERT_ALLDIMS_REDUCTION_OP_PATTERN(AtenOp, MhloOp)    \
  target.addIllegalOp<AtenOp>();                               \
  patterns.add<ConvertAtenAllDimsReductionOp<AtenOp, MhloOp>>( \
      typeConverter, context);
    INSERT_ALLDIMS_REDUCTION_OP_PATTERN(AtenAllOp, mhlo::AndOp)
    INSERT_ALLDIMS_REDUCTION_OP_PATTERN(AtenAnyOp, mhlo::OrOp)
    INSERT_ALLDIMS_REDUCTION_OP_PATTERN(AtenSumOp, mhlo::AddOp)
#undef INSERT_ALLDIMS_REDUCTION_OP_PATTERN
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

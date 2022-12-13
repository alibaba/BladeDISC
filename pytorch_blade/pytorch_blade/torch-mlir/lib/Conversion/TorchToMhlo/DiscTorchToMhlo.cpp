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

#include "torch-mlir/Conversion/MhloPasses.h"

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/utils/hlo_utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

#include "lib/Conversion/TorchToMhlo/MhloLegalizeUtils.h"
#include "stablehlo/dialect/ChloOps.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

#include <unordered_set>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::TorchConversion;

namespace {

static constexpr size_t kMhloDimSizeBits = 32;
LogicalResult BroadcastTensorRanks(
    PatternRewriter& rewriter,
    Operation* op,
    mlir::Value& self,
    mlir::Value& other) {
  auto selfTy = self.getType().template dyn_cast<RankedTensorType>();
  auto otherTy = other.getType().template dyn_cast<RankedTensorType>();
  auto selfRank = selfTy.getRank();
  auto otherRank = otherTy.getRank();
  if (selfRank == 0 || otherRank == 0)
    return success();
  if (selfRank > otherRank) {
    auto inputUnsqzDims =
        llvm::to_vector<4>(llvm::seq<int64_t>(0, selfRank - otherRank));
    auto unsqzInfo = mhlo::unsqueezeTensor(
        rewriter, op, other, inputUnsqzDims, kMhloDimSizeBits);
    if (failed(unsqzInfo))
      return failure();
    other = *unsqzInfo;
  } else if (otherRank > selfRank) {
    auto inputUnsqzDims =
        llvm::to_vector<4>(llvm::seq<int64_t>(0, otherRank - selfRank));
    auto unsqzInfo = mhlo::unsqueezeTensor(
        rewriter, op, self, inputUnsqzDims, kMhloDimSizeBits);
    if (failed(unsqzInfo))
      return failure();
    self = *unsqzInfo;
  }
  return success();
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
} // namespace

namespace {
template <>
LogicalResult ConvertAtenOp<OperatorOp>::matchAndRewrite(
    OperatorOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  Location loc = op.getLoc();
  auto name = op.name();
  auto outTy = getTypeConverter()
                   ->convertType(op.getResult(0).getType())
                   .dyn_cast<mlir::RankedTensorType>();
  if ("aten.einsum" == name) {
    SmallVector<Value> torchTensors;
    if (!getListConstructElements(op.getOperand(1), torchTensors))
      return rewriter.notifyMatchFailure(
          op, "tensors should come from a PrimListConstructOp");

    if (torchTensors.size() > 2)
      return rewriter.notifyMatchFailure(
          op, "aten::einsum with more than 2 inputs are not supported yet");

    SmallVector<Value> builtinTensors = Torch::getTypeConvertedValues(
        rewriter, op->getLoc(), getTypeConverter(), torchTensors);

    std::string equation;
    if (!matchPattern(op.getOperand(0), m_TorchConstantStr(equation)))
      return rewriter.notifyMatchFailure(op, "unknown equation");
    // TODO: an equation with "..." may require for implicit broadcast
    // and is not supported a.t.m.
    if (equation.find("...") != std::string::npos)
      return rewriter.notifyMatchFailure(
          op, "unsupported equation for aten::einsum");

    Value lhs = builtinTensors[0];
    Value rhs = builtinTensors[1];
    auto lhsTy = lhs.getType().dyn_cast<mlir::RankedTensorType>();
    auto rhsTy = rhs.getType().dyn_cast<mlir::RankedTensorType>();
    if (lhsTy.getElementType() != outTy.getElementType()) {
      lhs = rewriter.create<mlir::mhlo::ConvertOp>(
          loc, lhs, outTy.getElementType());
    }
    if (rhsTy.getElementType() != outTy.getElementType()) {
      rhs = rewriter.create<mlir::mhlo::ConvertOp>(
          loc, rhs, outTy.getElementType());
    }
    rewriter.replaceOpWithNewOp<mhlo::EinsumOp>(
        op,
        outTy,
        lhs,
        rhs,
        mlir::StringAttr::get(rewriter.getContext(), equation));
    return success();
  } else if ("prims.broadcast_in_dim" == name) {
    auto shape = op.getOperand(1);
    SmallVector<Value, 4> dimSizes;
    getListConstructElements(shape, dimSizes);
    // BladeDISC use i32 as shape
    std::for_each(dimSizes.begin(), dimSizes.end(), [&](Value& dSize) {
      dSize = rewriter.create<ToI64Op>(loc, dSize).getResult();
      // dimSize: cast i64 -> i32
      dSize =
          rewriter.create<arith::TruncIOp>(loc, rewriter.getI32Type(), dSize);
      return dSize;
    });

    SmallVector<int64_t> inputDims;
    if (!matchPattern(op.getOperand(2), m_TorchConstantIntList(inputDims))) {
      return rewriter.notifyMatchFailure(op, "non-int dim list unsupported");
    }
    SmallVector<Value> torchTensors{op.getOperand(0)};
    auto builtinTensors = Torch::getTypeConvertedValues(
        rewriter, op->getLoc(), getTypeConverter(), torchTensors);
    auto mhloShape =
        rewriter.create<mlir::tensor::FromElementsOp>(loc, dimSizes);
    auto result = rewriter.replaceOpWithNewOp<mhlo::DynamicBroadcastInDimOp>(
        op,
        outTy,
        builtinTensors[0],
        mhloShape,
        rewriter.getI64TensorAttr(inputDims));
    return success();
  }

  return failure();
}
} // namespace

namespace {
template <typename AtenOpT, typename ChloOpT>
class ConvertAtenBinaryBroadcastOp : public OpConversionPattern<AtenOpT> {
 public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult matchAndRewrite(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value lhs = adaptor.getOperands()[0];
    auto lhsTy = lhs.getType().cast<TensorType>();
    Value rhs = adaptor.getOperands()[1];
    auto rhsTy = rhs.getType().cast<TensorType>();

    if (!lhsTy || !rhsTy)
      return op.emitError("only Tensor types supported in MHLO");

    auto lhsElemTy = lhsTy.getElementType();
    auto rhsElemTy = rhsTy.getElementType();

    if (lhsElemTy != rhsElemTy)
      return op.emitError("input data types mismatched");

    rewriter.replaceOpWithNewOp<ChloOpT>(
        op,
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()),
        lhs,
        rhs,
        /*broadcast_attr*/ nullptr);
    return success();
  }
};
} // namespace

// ConvertAtenUnaryConvertOp legalize genearl unary ops into Mhlo ConverOp
namespace {
template <typename AtenOpT>
class ConvertAtenUnaryConvertOp : public OpConversionPattern<AtenOpT> {
 public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult matchAndRewrite(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<mhlo::ConvertOp>(
        op,
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getType()),
        adaptor.self());
    return success();
  }
};
} // namespace

// These unary op legalizations are identical for floating-point
// or quantized types
namespace {
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
} // namespace

namespace {
template <typename AtenOpT>
class ConvertAtenExtractOp : public OpConversionPattern<AtenOpT> {
 public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult matchAndRewrite(
      AtenOpT op,
      OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto input = adaptor.getOperands()[0];
    auto inpTy = input.getType().template dyn_cast<RankedTensorType>();
    if (!inpTy)
      return op.emitError("only RankedTensorType is supported");

    auto outTy = OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
        op.getType());
    auto elemTy = inpTy.getElementType();

    if (outTy != elemTy) {
      auto output =
          rewriter.create<tensor::ExtractOp>(op.getLoc(), elemTy, input);

      bool toWider =
          outTy.getIntOrFloatBitWidth() > elemTy.getIntOrFloatBitWidth();
      if (elemTy.isIntOrIndex()) {
        if (toWider) {
          rewriter.replaceOpWithNewOp<arith::ExtSIOp>(op, outTy, output);
        } else {
          rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, outTy, output);
        }
      } else {
        if (toWider) {
          rewriter.replaceOpWithNewOp<arith::ExtFOp>(op, outTy, output);
        } else {
          rewriter.replaceOpWithNewOp<arith::TruncFOp>(op, outTy, output);
        }
      }
    } else {
      rewriter.replaceOpWithNewOp<tensor::ExtractOp>(op, outTy, input);
    }
    return success();
  }
};
} // namespace

namespace {
template <>
LogicalResult ConvertAtenOp<AtenArangeOp>::matchAndRewrite(
    AtenArangeOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto end = adaptor.end();
  if (!end.getType().isIntOrIndex())
    return rewriter.notifyMatchFailure(op, "non-int end unsupported");

  if (!op.dtype().getType().isa<Torch::NoneType>()) {
    int64_t dtypeInt;
    if (!matchPattern(op.dtype(), m_TorchConstantInt(&dtypeInt)))
      return rewriter.notifyMatchFailure(op, "non-const dtype unsupported");

    Type resDtype = getTypeForScalarType(
        op.getContext(), (torch_upstream::ScalarType)dtypeInt);
    if (resDtype.isIntOrIndex())
      return rewriter.notifyMatchFailure(
          op, "non-int arange dtype unsupported");
  }

  auto iotaShape = rewriter.create<mlir::tensor::FromElementsOp>(
      op->getLoc(), ArrayRef<Value>{end});
  rewriter.replaceOpWithNewOp<mhlo::DynamicIotaOp>(
      op, getTypeConverter()->convertType(op.getType()), iotaShape, /*dim*/ 0);
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenCopyOp>::matchAndRewrite(
    AtenCopyOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  Location loc = op.getLoc();
  Value input = adaptor.src();
  auto inputTy = input.getType().template dyn_cast<RankedTensorType>();
  if (!inputTy) {
    return op.emitError("only RankedTensorType is supported");
  }

  rewriter.replaceOpWithNewOp<mhlo::ConvertOp>(
      op, getTypeConverter()->convertType(op.getType()), input);
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
    return op.emitError("only RankedTensorType is supported");
  }

  Value zero = chlo::getConstantLike(rewriter, loc, 0.0, input);
  Value six = chlo::getConstantLike(rewriter, loc, 6.0, input);
  Value relu = rewriter.create<mhlo::MaxOp>(loc, inputTy, input, zero);
  rewriter.replaceOpWithNewOp<mhlo::MinOp>(
      op, getTypeConverter()->convertType(op.getType()), relu, six);
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenWhereSelfOp>::matchAndRewrite(
    AtenWhereSelfOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  Value self = adaptor.self();
  Value cond = adaptor.condition();
  Value other = adaptor.other();

  if (failed(BroadcastTensorRanks(rewriter, op, self, cond)))
    return op.emitError("failed broadcast self and condition ranks");

  if (failed(BroadcastTensorRanks(rewriter, op, other, cond)))
    return op.emitError("failed broadcast other and condition ranks");

  rewriter.replaceOpWithNewOp<chlo::BroadcastSelectOp>(
      op,
      getTypeConverter()->convertType(op.getType()),
      ArrayRef<Value>{cond, self, other});
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenTensorOp>::matchAndRewrite(
    AtenTensorOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto listNode = op.data().getDefiningOp();
  if (!isa<PrimListConstructOp>(listNode))
    return op.emitError(
        "input list is not generated by torch.prim.ListConstruct");

  auto outTy = getTypeConverter()
                   ->convertType(op.getType())
                   .dyn_cast<RankedTensorType>();
  SmallVector<Value, 4> inputs;
  for (auto operand : listNode->getOperands()) {
    auto inpTy = operand.getType();
    if (isa<Torch::BoolType>(inpTy)) {
      operand = rewriter.create<ToI1Op>(op.getLoc(), operand);
    } else if (isa<Torch::IntType>(inpTy)) {
      operand = rewriter.create<ToI64Op>(op.getLoc(), operand);
    } else if (isa<Torch::FloatType>(inpTy)) {
      operand = rewriter.create<ToF64Op>(op.getLoc(), operand);
      if (!outTy.getElementType().isF64()) {
        operand = rewriter.create<arith::TruncFOp>(
            op.getLoc(), outTy.getElementType(), operand);
      }
    }
    inputs.push_back(operand);
  }

  rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(op, outTy, inputs);
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenTensorFloatOp>::matchAndRewrite(
    AtenTensorFloatOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto outTy = getTypeConverter()
                   ->convertType(op.getType())
                   .dyn_cast<RankedTensorType>();
  auto val = adaptor.t();
  if (!outTy.getElementType().isF64()) {
    val = rewriter.create<arith::TruncFOp>(
        op.getLoc(), outTy.getElementType(), val);
  }
  rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(
      op, outTy, ArrayRef<Value>{val});
  return success();
}

template <>
LogicalResult ConvertAtenOp<AtenTensorBoolOp>::matchAndRewrite(
    AtenTensorBoolOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(
      op,
      getTypeConverter()->convertType(op.getType()),
      ArrayRef<Value>{adaptor.t()});
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
LogicalResult ConvertAtenOp<AtenFloatScalarOp>::matchAndRewrite(
    AtenFloatScalarOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  rewriter.replaceOpWithNewOp<arith::SIToFPOp>(
      op, getTypeConverter()->convertType(op.getType()), adaptor.a());
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
  rewriter.replaceOpWithNewOp<mhlo::DivOp>(
      op, getTypeConverter()->convertType(op.getType()), one, addVal);
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

  Value dim;
  int64_t dimInt;
  if (matchPattern(op.dim(), m_TorchConstantInt(&dimInt))) {
    dimInt = toPositiveDim(dimInt, selfType.getRank());
    dim = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), dimInt);
  } else {
    dim = rewriter.create<arith::IndexCastOp>(
        op.getLoc(), rewriter.getIndexType(), adaptor.dim());
  }

  auto dimSize = rewriter.create<tensor::DimOp>(
      op.getLoc(), rewriter.getIndexType(), adaptor.self(), dim);

  rewriter.replaceOpWithNewOp<arith::IndexCastOp>(
      op, getTypeConverter()->convertType(op.getType()), dimSize);

  return success();
}

// Convert a Aten::Silu to HLO
// Silu(x) = x * Sigmoid(x)
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
  rewriter.replaceOpWithNewOp<mhlo::MulOp>(
      op, getTypeConverter()->convertType(op.getType()), grad_output, result);
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
      rewriter.getI64TensorAttr(dimListInt));

  return success();
}

// AtenUniformOp
template <>
LogicalResult ConvertAtenOp<AtenUniformOp>::matchAndRewrite(
    AtenUniformOp op,
    OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto inputTy = adaptor.self().getType().template cast<RankedTensorType>();
  auto loc = op.getLoc();
  if (!inputTy) {
    op.emitError("input should be ranked tensor type.");
  }

  auto inputShapeInfo =
      mhlo::getDimSizesOfTensor(rewriter, op, adaptor.self(), kMhloDimSizeBits);
  if (failed(inputShapeInfo)) {
    return rewriter.notifyMatchFailure(
        op, "failed to get dimension sizes of the input");
  }

  SmallVector<Value, 4> dimSizes = *inputShapeInfo;
  if (dimSizes.size() == 0) {
    return op.emitError("the input rank is 0");
  }
  auto mhloShape =
      rewriter.create<tensor::FromElementsOp>(op.getLoc(), dimSizes);
  double fromDoubleValue, toDoubleValue;
  if (!matchPattern(op.from(), m_TorchConstantFloat(&fromDoubleValue))) {
    op.emitError("operand #1 should be scalar");
  }
  if (!matchPattern(op.to(), m_TorchConstantFloat(&toDoubleValue))) {
    op.emitError("operand #2 should be scalar");
  }
  Value fromTensor = rewriter.create<mhlo::ConstantOp>(
      op.getLoc(),
      rewriter.getFloatAttr(inputTy.getElementType(), fromDoubleValue));
  Value toTensor = rewriter.create<mhlo::ConstantOp>(
      op.getLoc(),
      rewriter.getFloatAttr(inputTy.getElementType(), toDoubleValue));

  auto outType = getTypeConverter()
                     ->convertType(op.getType())
                     .template dyn_cast<TensorType>();
  rewriter.replaceOpWithNewOp<mhlo::RngOp>(
      op,
      inputTy,
      fromTensor,
      toTensor,
      mhloShape,
      mhlo::RngDistribution::UNIFORM);
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
        "only RankedTensorType is supported in Aten LeakyReLU op.");
  }

  double scaleValue;
  if (!matchPattern(negativeSlope, m_TorchConstantFloat(&scaleValue)))
    return op->emitError(
        "currently only scalar constants are supported for "
        "negative_slope in MHLO operation");

  Value zeroVal = chlo::getConstantLike(rewriter, loc, 0.0, input);
  Value scaleVal = chlo::getConstantLike(rewriter, loc, scaleValue, input);

  Value leakyActivationVal = rewriter.create<mhlo::MulOp>(
      loc, getTypeConverter()->convertType(op.getType()), input, scaleVal);

  Value compareGtZero = rewriter.create<mhlo::CompareOp>(
      loc, input, zeroVal, mhlo::ComparisonDirection::GT);

  rewriter.replaceOpWithNewOp<mhlo::SelectOp>(
      op,
      getTypeConverter()->convertType(op.getType()),
      compareGtZero,
      input,
      leakyActivationVal);
  return success();
}
} // namespace

namespace {
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
    Block& block = valueReduceOp.getBody().emplaceBlock();
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
    Block& block = indicesReduceOp.getBody().emplaceBlock();

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

  auto inputShapeInfo =
      mhlo::getDimSizesOfTensor(rewriter, op, input, kMhloDimSizeBits);
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
} // namespace
// -----------------------------------------------------------------------------
// TorchToMhlo Pass
// -----------------------------------------------------------------------------

namespace {
class DiscConvertTorchToMhlo
    : public TorchConversion::DiscConvertTorchToMhloBase<
          DiscConvertTorchToMhlo> {
 public:
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<chlo::ChloDialect>();
    registry.insert<mhlo::MhloDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<Torch::TorchDialect>();
    torch::TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<
        chlo::ChloDialect,
        mhlo::MhloDialect,
        tensor::TensorDialect,
        arith::ArithDialect,
        Torch::TorchDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);
    auto opIsDynamicallyLegal = [&](OperatorOp op) {
      static std::unordered_set<std::string> illegalSet{
          "aten.einsum",
          "prims.broadcast_in_dim",
      };
      if (illegalSet.find(op.name().str()) != illegalSet.end()) {
        return false;
      }
      return true;
    };

    // Won't mark OperatorOp as illegal, some custom operator may remain
    // unconverted.
    target.addDynamicallyLegalOp<OperatorOp>(opIsDynamicallyLegal);
    patterns.add<ConvertAtenOp<OperatorOp>>(typeConverter, context);

#define INSERT_UNARY_CONVERT_PATTERN(AtenOp) \
  target.addIllegalOp<AtenOp>();             \
  patterns.add<ConvertAtenUnaryConvertOp<AtenOp>>(typeConverter, context);
    INSERT_UNARY_CONVERT_PATTERN(AtenContiguousOp);
    INSERT_UNARY_CONVERT_PATTERN(AtenToDtypeOp);
    INSERT_UNARY_CONVERT_PATTERN(AtenToDtypeLayoutOp);
    INSERT_UNARY_CONVERT_PATTERN(AtenToPrimDeviceOp);
    INSERT_UNARY_CONVERT_PATTERN(AtenTypeAsOp);
#undef INSERT_UNARY_CONVERT_PATTERN

#define INSERT_UNARY_PATTERN(AtenOp, MhloOp) \
  target.addIllegalOp<AtenOp>();             \
  patterns.add<ConvertAtenUnaryOp<AtenOp, MhloOp>>(typeConverter, context);
    INSERT_UNARY_PATTERN(AtenBitwiseNotOp, mhlo::NotOp)
    INSERT_UNARY_PATTERN(AtenNegOp, mhlo::NegOp)
    INSERT_UNARY_PATTERN(AtenFloorOp, mhlo::FloorOp)
    INSERT_UNARY_PATTERN(AtenCeilOp, mhlo::CeilOp)
    INSERT_UNARY_PATTERN(AtenItemOp, tensor::ExtractOp)
    INSERT_UNARY_PATTERN(AtenCopyOp, mhlo::CopyOp)
    INSERT_UNARY_PATTERN(AtenCosOp, mhlo::CosineOp)
    INSERT_UNARY_PATTERN(AtenSinOp, mhlo::SineOp)
#undef INSERT_UNARY_PATTERN

#define INSERT_EXTRACT_PATTERN(AtenOp) \
  target.addIllegalOp<AtenOp>();       \
  patterns.add<ConvertAtenExtractOp<AtenOp>>(typeConverter, context);
    INSERT_EXTRACT_PATTERN(AtenScalarImplicitOp)
    INSERT_EXTRACT_PATTERN(AtenIntTensorOp)
    INSERT_EXTRACT_PATTERN(AtenItemOp)
#undef INSERT_EXTRACT_PATTERN

#define INSERT_ATENOP_PATTERN(AtenOp) \
  target.addIllegalOp<AtenOp>();      \
  patterns.add<ConvertAtenOp<AtenOp>>(typeConverter, context)
    INSERT_ATENOP_PATTERN(AtenArangeOp);
    INSERT_ATENOP_PATTERN(AtenCopyOp);
    INSERT_ATENOP_PATTERN(AtenLeakyReluOp);
    INSERT_ATENOP_PATTERN(AtenRelu6Op);
    INSERT_ATENOP_PATTERN(AtenSiluOp);
    INSERT_ATENOP_PATTERN(AtenSizeIntOp);
    INSERT_ATENOP_PATTERN(AtenSigmoidOp);
    INSERT_ATENOP_PATTERN(AtenGeluBackwardOp);
    INSERT_ATENOP_PATTERN(AtenEmptyMemoryFormatOp);
    INSERT_ATENOP_PATTERN(AtenDropoutOp);
    INSERT_ATENOP_PATTERN(TensorStaticInfoCastOp);
    INSERT_ATENOP_PATTERN(AtenFlipOp);
    INSERT_ATENOP_PATTERN(AtenUniformOp);
    INSERT_ATENOP_PATTERN(AtenTensorOp);
    INSERT_ATENOP_PATTERN(AtenTensorBoolOp);
    INSERT_ATENOP_PATTERN(AtenTensorFloatOp);
    INSERT_ATENOP_PATTERN(AtenTensorIntOp);
    INSERT_ATENOP_PATTERN(AtenFloatScalarOp);
    INSERT_ATENOP_PATTERN(AtenMaxDimOp);
    INSERT_ATENOP_PATTERN(AtenWhereSelfOp);
#undef INSERT_ATENOP_PATTERN

#define INSERT_BINARY_BROADCAST_PATTERN(AtenOp, MhloOp)       \
  target.addIllegalOp<AtenOp>();                              \
  patterns.add<ConvertAtenBinaryBroadcastOp<AtenOp, MhloOp>>( \
      typeConverter, context)
    INSERT_BINARY_BROADCAST_PATTERN(AtenMaximumOp, chlo::BroadcastMaxOp);
    INSERT_BINARY_BROADCAST_PATTERN(
        AtenPowTensorTensorOp, chlo::BroadcastPowOp);
    INSERT_BINARY_BROADCAST_PATTERN(AtenMinimumOp, chlo::BroadcastMinOp);
    INSERT_BINARY_BROADCAST_PATTERN(Aten__And__TensorOp, chlo::BroadcastAndOp);
    INSERT_BINARY_BROADCAST_PATTERN(
        AtenBitwiseAndTensorOp, chlo::BroadcastAndOp);
#undef INSERT_BINARY_BROADCAST_PATTERN

    if (failed(applyPartialConversion(
            getOperation(), target, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::torch::TorchConversion::
    createDiscConvertTorchToMhloPass() {
  return std::make_unique<DiscConvertTorchToMhlo>();
}

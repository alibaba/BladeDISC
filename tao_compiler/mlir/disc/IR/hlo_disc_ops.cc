/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// This file defines the operations used in the DISC RAL dialect.

#include "mlir/disc/IR/hlo_disc_ops.h"

#include <cfenv>
#include <cmath>
#include <iostream>
#include <mutex>
#include <unordered_map>

#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/disc/IR/custom_call_base.h"
#include "mlir/disc/IR/hlo_disc_enums.cc.inc"

namespace mlir {
namespace mhlo_disc {

using llvm::StringRef;

template <typename T>
static LogicalResult Verify(T op) {
  return success();
}

//===----------------------------------------------------------------------===//
// MHLO Dialect Hooks
//===----------------------------------------------------------------------===//

Operation* MhloDiscDialect::materializeConstant(OpBuilder& builder,
                                                Attribute value, Type type,
                                                Location loc) {
  auto elementsAttr = value.dyn_cast<ElementsAttr>();
  // HLO dialect constants only support ElementsAttr unlike standard dialect
  // constant which supports all attributes.
  if (!elementsAttr) return nullptr;
  // HLO dialect constants require the type of value and result to match.
  if (type != elementsAttr.getType()) return nullptr;
  Operation* op = builder.create<mhlo::ConstantOp>(loc, type, elementsAttr);
  return op;
}

//===----------------------------------------------------------------------===//
// mhlo disc Dialect Constructor
//===----------------------------------------------------------------------===//

MhloDiscDialect::MhloDiscDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context, TypeID::get<MhloDiscDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "mlir/disc/IR/hlo_disc_ops.cc.inc"

      >();
  context->loadDialect<tensor::TensorDialect>();
}

//===----------------------------------------------------------------------===//
// H2DOp
//===----------------------------------------------------------------------===//

LogicalResult H2DOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  return mlir::hlo::deriveShapeFromOperand(&builder, getOperation(),
                                           operands[0], &reifiedReturnShapes);
}

LogicalResult H2DOp::verify() { return Verify(*this); }

//===----------------------------------------------------------------------===//
// D2HOp
//===----------------------------------------------------------------------===//

LogicalResult D2HOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  return mlir::hlo::deriveShapeFromOperand(&builder, getOperation(),
                                           operands[0], &reifiedReturnShapes);
}

LogicalResult D2HOp::verify() { return Verify(*this); }

//===----------------------------------------------------------------------===//
// CustomCallOp
//===----------------------------------------------------------------------===//

LogicalResult CustomCallOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  CustomCallOp::Adaptor adaptor(operands);
  ValueRange args = adaptor.getArgs();
  StringRef target = getCallTargetName();
  auto reify_shapes_func =
      CustomCallRegistry::Global().FindReifyShapesFunc(target.str());
  if (!reify_shapes_func) {
    return emitOpError() << "custom call " << target << " is not supported.";
  }
  return reify_shapes_func(*this, builder, operands, reifiedReturnShapes);
}

LogicalResult CustomCallOp::verify() { return Verify(*this); }

//===----------------------------------------------------------------------===//
// FakeQuantOp
//===----------------------------------------------------------------------===//

template <typename T>
LogicalResult QuantVerify(T* op) {
  auto inputTy = op->getInput().getType().template dyn_cast<RankedTensorType>();
  auto scaleTy = op->getScale().getType().template dyn_cast<RankedTensorType>();
  auto zeroPointTy =
      op->getZeroPoint().getType().template dyn_cast<RankedTensorType>();
  auto resultTy =
      op->getResult().getType().template dyn_cast<RankedTensorType>();
  if (!inputTy || !scaleTy || !zeroPointTy || !resultTy)
    return op->emitOpError() << "only support ranked input.\n";
  if (inputTy.getShape() != resultTy.getShape())
    return op->emitOpError() << "input and result have mismatch shape.\n";
  if (scaleTy.getRank() != zeroPointTy.getRank())
    return op->emitOpError() << "scale and zero_point have mismatch rank.\n";
  auto axis = op->getAxis().template getValues<int64_t>();
  if (axis.size() != scaleTy.getRank())
    return op->emitOpError() << "num of quantized axes (len(axis)) is not "
                                "equal to the rank of scale tensor\n";
  return success();
}

LogicalResult FakeQuantOp::verify() { return QuantVerify(this); }

//===----------------------------------------------------------------------===//
// QuantizeOp
//===----------------------------------------------------------------------===//

LogicalResult QuantizeOp::verify() { return QuantVerify(this); }

OpFoldResult QuantizeOp::fold(ArrayRef<Attribute> operands) {
  auto val = operands[0].dyn_cast_or_null<DenseElementsAttr>();
  if (!val) {
    return {};
  }
  auto valType = val.getType();
  auto valElementType = getElementTypeOrSelf(valType);
  if (!valElementType.isF32()) {
    return {};
  }

  auto opType = getType();
  auto opElementType = getElementTypeOrSelf(opType);
  if (!opElementType.isa<IntegerType>()) {
    return {};
  }

  auto valShapedType = valType.cast<ShapedType>();
  if (!valShapedType.hasStaticShape()) {
    return {};
  }

  DenseElementsAttr scale = operands[1].dyn_cast_or_null<DenseElementsAttr>();
  DenseElementsAttr zeroPoint =
      operands[2].dyn_cast_or_null<DenseElementsAttr>();
  if (!scale || !zeroPoint) return {};
  ArrayRef<int64_t> scaleShape = scale.getType().getShape();
  ArrayRef<int64_t> zeroPointShape = zeroPoint.getType().getShape();
  // scale & zero_point must have the same shape
  if (scaleShape != zeroPointShape) {
    return {};
  }

  DenseIntElementsAttr axis = getAxisAttr();
  int64_t axisNum = axis.getNumElements();
  // only per-tensor/per-channel quantization is supported
  if (!(axisNum <= 1)) {
    return {};
  }

  int64_t scaleRank = scale.getType().getRank();
  int64_t zeroPointRank = zeroPoint.getType().getRank();
  if (!(axisNum == scaleRank && axisNum == zeroPointRank)) {
    // per-tensor quantization: axisNum==scaleRank==zeroPointRank==0
    // per-channel quantization: axisNum==scaleRank==zeroPointRank==1
    return {};
  }
  // For per-tensor quantization, its axis is empty. For convenience,
  // we use -1 to represent the axis dimension it point to.
  int64_t symbolAxis = -1;
  int64_t axisValue =
      axisNum == 0 ? symbolAxis : *axis.getValues<int64_t>().begin();
  int64_t valRank = valType.getRank();
  if (axisValue >= valRank) {
    // axisValue must point to a valid dimension of the input.
    return {};
  }

  int64_t scaleNum = scale.getNumElements();
  int64_t zeroPointNum = zeroPoint.getNumElements();
  ArrayRef<int64_t> valShape = valType.getShape();
  int64_t targetQuantInfoNum =
      axisValue == symbolAxis ? 1 : valShape[axisValue];
  if (!(targetQuantInfoNum == scaleNum && targetQuantInfoNum == zeroPointNum)) {
    // For per-tensor quantization, scale and zero_point only have one value.
    // For per-channel quantization, The length of scale and zero_point must be
    // equal to the length of the input dimension pointed to by axis
    return {};
  }

  int64_t numPerScale = 1;
  for (int64_t i = 0; i < valShape.size(); i++) {
    if (i > axisValue) {
      numPerScale *= valShape[i];
    }
  }

  auto roundToEven = [&](double val) -> double {
    auto curRoundMode = std::fegetround();
    std::fesetround(FE_TONEAREST);
    float roundVal = std::nearbyint(val);
    std::fesetround(curRoundMode);
    return roundVal;
  };

  int64_t valNum = val.getNumElements();
  auto valStart = val.getValues<APFloat>().begin();
  auto scaleStart = scale.getValues<APFloat>().begin();
  auto zeroPointStart = zeroPoint.getValues<APInt>().begin();
  int64_t quantMax = getQuantMax();
  int64_t quantMin = getQuantMin();
  bool useSigned = getUseSymmetric();
  RoundModeEnum roundMode = getRoundMode();
  double curValNum;
  double curScale;
  int64_t curZeroPoint;
  int64_t quantInfoIdx;
  int bitWidth = opElementType.getIntOrFloatBitWidth();

  int64_t bytesPerElem;
  if (bitWidth == 32) {
    bytesPerElem = sizeof(int32_t);
  } else if (bitWidth == 16) {
    bytesPerElem = sizeof(int16_t);
  } else if (bitWidth == 8) {
    bytesPerElem = sizeof(int8_t);
  } else {
    // only supports 32/16/8 bit quantization for now
    return {};
  }
  int64_t bytes = valNum * bytesPerElem;
  std::vector<uint8_t> buffer(bytes);

  // use double to store the quantized value first and static_cast
  // it to the target format later.
  std::vector<double> quantizedVal;
  quantizedVal.reserve(valNum);

  for (int64_t i = 0; i < valNum; i++) {
    curValNum = (*(valStart + i)).convertToDouble();
    quantInfoIdx = i / numPerScale % targetQuantInfoNum;
    curScale = (*(scaleStart + quantInfoIdx)).convertToDouble();
    int64_t curZeroPoint = (*(zeroPointStart + quantInfoIdx)).getSExtValue();
    double curValNumDiv = (curValNum / curScale) + curZeroPoint;
    double curValNumDivRound;
    if (roundMode == RoundModeEnum::RoundHalfToEven) {
      curValNumDivRound = roundToEven(curValNumDiv);
    } else if (roundMode == RoundModeEnum::RoundHalfAwayFromZero) {
      // round away from zero
      curValNumDivRound = std::round(curValNumDiv);
    } else {
      return {};
    }
    curValNum =
        std::clamp(curValNumDivRound, (double)quantMin, double(quantMax));
    quantizedVal.push_back(curValNum);
  }
  auto newOutput = DenseElementsAttr();
  auto outType = getType().cast<ShapedType>();
  if (bitWidth == 32) {
    if (useSigned) {
      auto data = (int32_t*)buffer.data();
      for (int64_t i = 0; i < valNum; i++) {
        data[i] = static_cast<int32_t>(quantizedVal[i]);
      }
      newOutput =
          DenseElementsAttr::get(outType, llvm::makeArrayRef(data, valNum));
    } else {
      auto data = (uint32_t*)buffer.data();
      for (int64_t i = 0; i < valNum; i++) {
        data[i] = static_cast<uint32_t>(quantizedVal[i]);
      }
      newOutput =
          DenseElementsAttr::get(outType, llvm::makeArrayRef(data, valNum));
    }
  } else if (bitWidth == 16) {
    if (useSigned) {
      auto data = (int16_t*)buffer.data();
      for (int64_t i = 0; i < valNum; i++) {
        data[i] = static_cast<int16_t>(quantizedVal[i]);
      }
      newOutput =
          DenseElementsAttr::get(outType, llvm::makeArrayRef(data, valNum));
    } else {
      auto data = (uint16_t*)buffer.data();
      for (int64_t i = 0; i < valNum; i++) {
        data[i] = static_cast<uint16_t>(quantizedVal[i]);
      }
      newOutput =
          DenseElementsAttr::get(outType, llvm::makeArrayRef(data, valNum));
    }
  } else if (bitWidth == 8) {
    if (useSigned) {
      auto data = (int8_t*)buffer.data();
      for (int64_t i = 0; i < valNum; i++) {
        data[i] = static_cast<int8_t>(quantizedVal[i]);
      }
      newOutput =
          DenseElementsAttr::get(outType, llvm::makeArrayRef(data, valNum));
    } else {
      auto data = (uint8_t*)buffer.data();
      for (int64_t i = 0; i < valNum; i++) {
        data[i] = static_cast<uint8_t>(quantizedVal[i]);
      }
      newOutput =
          DenseElementsAttr::get(outType, llvm::makeArrayRef(data, valNum));
    }
  }

  return newOutput;
}

//===----------------------------------------------------------------------===//
// DequantizeOp
//===----------------------------------------------------------------------===//

LogicalResult DequantizeOp::verify() { return QuantVerify(this); }

//===----------------------------------------------------------------------===//
// QuantizedDotGeneralOp
//===----------------------------------------------------------------------===//

template <typename T>
LogicalResult CommonVerifyForQuantizedComputeIntensiveOp(T* op) {
  auto inputTy = op->getInput().getType().template dyn_cast<RankedTensorType>();
  auto weightTy =
      op->getWeight().getType().template dyn_cast<RankedTensorType>();
  auto resultTy =
      op->getResult().getType().template dyn_cast<RankedTensorType>();

  if (!inputTy || !weightTy || !resultTy ||
      inputTy.getRank() != weightTy.getRank() ||
      inputTy.getRank() != resultTy.getRank()) {
    return op->emitOpError()
           << "input, weight and result should have the same rank.\n";
  }

  auto inputScaleTy =
      op->getInputScale().getType().template dyn_cast<RankedTensorType>();
  auto inputZeroPointTy =
      op->getInputZeroPoint().getType().template dyn_cast<RankedTensorType>();
  if (!inputScaleTy || !inputZeroPointTy || inputScaleTy.getRank() != 0 ||
      inputZeroPointTy.getRank() != 0) {
    return op->emitOpError() << "input_scale and input_zero_point only support "
                                "per-tensor quantization\n";
  }

  auto resultScaleTy =
      op->getResultScale().getType().template dyn_cast<RankedTensorType>();
  auto resultZeroPointTy =
      op->getResultZeroPoint().getType().template dyn_cast<RankedTensorType>();
  if (!resultScaleTy || !resultZeroPointTy || resultScaleTy.getRank() != 0 ||
      resultZeroPointTy.getRank() != 0) {
    return op->emitOpError() << "result_scale and result_zero_point only "
                                "support per-tensor quantization\n";
  }

  auto weightScaleTy =
      op->getWeightScale().getType().template dyn_cast<RankedTensorType>();
  auto weightZeroPointTy =
      op->getWeightZeroPoint().getType().template dyn_cast<RankedTensorType>();
  if (!weightScaleTy || !weightZeroPointTy ||
      weightScaleTy.getShape() != weightZeroPointTy.getShape()) {
    return op->emitOpError()
           << "weight_scale and weight_zero_point have mismatch shape\n";
  }
  auto axis = op->getAxis().template getValues<int64_t>();
  if (axis.size() != weightScaleTy.getRank())
    return op->emitOpError() << "num of quantized axes (len(axis)) is not "
                                "equal to the rank of weight_scale tensor\n";
  return success();
}

LogicalResult QuantizedDotGeneralOp::verify() {
  return CommonVerifyForQuantizedComputeIntensiveOp(this);
}

LogicalResult QuantizedDotGeneralOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  auto lhsType = getInput().getType().dyn_cast<ShapedType>();
  auto rhsType = getWeight().getType().dyn_cast<ShapedType>();
  if (!lhsType || !rhsType) {
    return failure();
  }

  Adaptor adaptor(operands);
  auto dimNumbers = getDotDimensionNumbers();
  SmallVector<Value> dimensions;
  for (const int64_t lhsDim : dimNumbers.getLhsBatchingDimensions()) {
    dimensions.push_back(
        builder.create<tensor::DimOp>(getLoc(), adaptor.getInput(), lhsDim));
  }

  for (int64_t i = 0; i < lhsType.getRank(); i++) {
    if (!llvm::is_contained(dimNumbers.getLhsContractingDimensions(), i) &&
        !llvm::is_contained(dimNumbers.getLhsBatchingDimensions(), i)) {
      dimensions.push_back(
          builder.create<tensor::DimOp>(getLoc(), adaptor.getInput(), i));
    }
  }
  for (int64_t i = 0; i < rhsType.getRank(); i++) {
    if (!llvm::is_contained(dimNumbers.getRhsContractingDimensions(), i) &&
        !llvm::is_contained(dimNumbers.getRhsBatchingDimensions(), i)) {
      dimensions.push_back(
          builder.create<tensor::DimOp>(getLoc(), adaptor.getWeight(), i));
    }
  }
  reifiedReturnShapes.push_back(
      builder.create<tensor::FromElementsOp>(getLoc(), dimensions));
  return success();
}

//===----------------------------------------------------------------------===//
// QuantizedDynamicConvOp
//===----------------------------------------------------------------------===//

LogicalResult QuantizedDynamicConvOp::verify() {
  return CommonVerifyForQuantizedComputeIntensiveOp(this);
}

//===----------------------------------------------------------------------===//
// Mostly copy from the implementation of mhlo::DynamicConvOp
//===----------------------------------------------------------------------===//

// Returns a new scalar integer value having type `type`. Here `type` must be
// an integer or index type.
Value maybeCastTo(OpBuilder& b, Location loc, Value value, Type type) {
  if (type == value.getType()) return value;
  if (!type.isIndex() && !value.getType().isIndex()) {
    // in case of i32 -> i64 or vice versa
    Value casted = b.create<arith::IndexCastOp>(loc, b.getIndexType(), value);
    return b.create<arith::IndexCastOp>(loc, type, casted);
  }
  return b.create<arith::IndexCastOp>(loc, type, value);
}

template <typename Op>
LogicalResult ConvReifyReturnTypeImpl(
    Op* op, OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes,
    const SmallVector<Value>& spatial_padding_values, Type shape_scalar_type) {
  typename Op::Adaptor adaptor(operands);
  Value lhs = adaptor.getInput();
  Value rhs = adaptor.getWeight();

  RankedTensorType lhs_type = lhs.getType().dyn_cast<RankedTensorType>();
  RankedTensorType rhs_type = rhs.getType().dyn_cast<RankedTensorType>();
  // Not support unranked type a.t.m.
  if (!lhs_type || !rhs_type) return failure();

  Location loc = op->getLoc();

  auto to_shape_scalar_type = [&](Value v) {
    return maybeCastTo(builder, loc, v, shape_scalar_type);
  };

  auto dimension_numbers = op->getDimensionNumbers();
  int64_t input_batch_dimension = dimension_numbers.getInputBatchDimension();
  int64_t kernel_output_feature_dimension =
      dimension_numbers.getKernelOutputFeatureDimension();
  auto input_spatial_dimensions_attr =
      dimension_numbers.getInputSpatialDimensions();
  auto kernel_spatial_dimensions_attr =
      dimension_numbers.getKernelSpatialDimensions();
  auto output_spatial_dimensions_attr =
      dimension_numbers.getOutputSpatialDimensions();

  SmallVector<Value, 4> shape_values(output_spatial_dimensions_attr.size() + 2);

  // batch dim = lhs-batch-dim / batch_group_count
  Value lhs_batch_dim = to_shape_scalar_type(
      builder.create<tensor::DimOp>(loc, lhs, input_batch_dimension));
  Value batch_group_count = to_shape_scalar_type(
      builder.create<arith::ConstantIndexOp>(loc, op->getBatchGroupCount()));
  Value batch_dim = to_shape_scalar_type(
      builder.create<arith::DivSIOp>(loc, lhs_batch_dim, batch_group_count));
  int64_t output_batch_dimension = dimension_numbers.getOutputBatchDimension();
  shape_values[output_batch_dimension] = batch_dim;

  // Output's feature dim is the same with kernel's output feature dim.
  Value feature_dim = to_shape_scalar_type(
      builder.create<tensor::DimOp>(loc, rhs, kernel_output_feature_dimension));
  int64_t output_feature_dimension =
      dimension_numbers.getOutputFeatureDimension();
  shape_values[output_feature_dimension] = feature_dim;

  Optional<DenseIntElementsAttr> window_strides_attr = op->getWindowStrides();
  Optional<DenseIntElementsAttr> lhs_dilation_attr = op->getLhsDilation();
  Optional<DenseIntElementsAttr> rhs_dilation_attr = op->getRhsDilation();

  Value one =
      to_shape_scalar_type(builder.create<arith::ConstantIndexOp>(loc, 1));
  for (uint64_t i = 0; i < output_spatial_dimensions_attr.size(); i++) {
    // effective_input_value =
    //    (input_size - 1) * input_dilation + 1 + padding_left + padding_right
    Value effective_input_value =
        to_shape_scalar_type(builder.create<tensor::DimOp>(
            loc, lhs, input_spatial_dimensions_attr[i]));
    // Dilation.
    if (lhs_dilation_attr) {
      Value input_dilation =
          to_shape_scalar_type(builder.create<arith::ConstantIndexOp>(
              loc, lhs_dilation_attr.value().getValues<int64_t>()[i]));
      effective_input_value = builder.create<arith::AddIOp>(
          loc,
          builder.create<arith::MulIOp>(
              loc,
              builder.create<arith::SubIOp>(loc, effective_input_value, one),
              input_dilation),
          one);
    }

    // Padding.
    if (!spatial_padding_values.empty()) {
      Value padding_left = spatial_padding_values[i * 2];
      Value padding_right = spatial_padding_values[i * 2 + 1];
      effective_input_value = builder.create<arith::AddIOp>(
          loc, effective_input_value,
          builder.create<arith::AddIOp>(loc, padding_left, padding_right));
    }

    // effective_kernel_size = (kernel_size - 1) * dilation + 1
    Value effective_kernel_size_value =
        to_shape_scalar_type(builder.create<tensor::DimOp>(
            loc, rhs, kernel_spatial_dimensions_attr[i]));
    if (rhs_dilation_attr) {
      Value kernel_dilation =
          to_shape_scalar_type(builder.create<arith::ConstantIndexOp>(
              loc, rhs_dilation_attr.value().getValues<int64_t>()[i]));
      effective_kernel_size_value = builder.create<arith::AddIOp>(
          loc, one,
          builder.create<arith::MulIOp>(
              loc, kernel_dilation,
              builder.create<arith::SubIOp>(loc, effective_kernel_size_value,
                                            one)));
    }

    // output_size =
    //     (effective_input_value - effective_kernel_size_value) / stride + 1
    Value output_dim_value = builder.create<arith::SubIOp>(
        loc, effective_input_value, effective_kernel_size_value);
    if (window_strides_attr) {
      Value stride_value =
          to_shape_scalar_type(builder.create<arith::ConstantIndexOp>(
              loc, window_strides_attr.value().getValues<int64_t>()[i]));
      output_dim_value =
          builder.create<arith::DivSIOp>(loc, output_dim_value, stride_value);
    }
    output_dim_value =
        builder.create<arith::AddIOp>(loc, output_dim_value, one);
    shape_values[output_spatial_dimensions_attr[i]] = output_dim_value;
  }
  Value output_shape =
      builder.create<tensor::FromElementsOp>(loc, shape_values);
  reifiedReturnShapes.push_back(output_shape);
  return success();
}

LogicalResult QuantizedDynamicConvOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  QuantizedDynamicConvOp::Adaptor adaptor(operands);
  Value d_padding = adaptor.getDPadding();

  RankedTensorType padding_type =
      d_padding.getType().dyn_cast<RankedTensorType>();
  // Not support unranked type a.t.m.
  if (!padding_type) return failure();

  Location loc = this->getLoc();
  Type shape_scalar_type = padding_type.getElementType();
  auto to_shape_scalar_type = [&](Value v) {
    return maybeCastTo(builder, loc, v, shape_scalar_type);
  };

  SmallVector<Value> spatial_padding_values;
  auto dimension_numbers = this->getDimensionNumbers();
  auto input_spatial_dimensions_attr =
      dimension_numbers.getInputSpatialDimensions();
  int64_t padding_num = input_spatial_dimensions_attr.size() * 2;
  for (int64_t i = 0; i < padding_num; i++) {
    Value offset = builder.create<arith::ConstantIndexOp>(loc, i);
    Value pad_value = to_shape_scalar_type(
        builder.create<tensor::ExtractOp>(loc, d_padding, offset));
    spatial_padding_values.push_back(pad_value);
  }

  return ConvReifyReturnTypeImpl(this, builder, operands, reifiedReturnShapes,
                                 spatial_padding_values, shape_scalar_type);
}

//===----------------------------------------------------------------------===//
// SparseReshapeOp
//===----------------------------------------------------------------------===//

LogicalResult SparseReshapeOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  SparseReshapeOp::Adaptor adaptor(operands);
  auto input_indices_type =
      adaptor.getInputIndices().getType().dyn_cast<RankedTensorType>();
  auto input_shape_type =
      adaptor.getInputShape().getType().dyn_cast<RankedTensorType>();
  auto new_shape_type =
      adaptor.getNewShape().getType().dyn_cast<RankedTensorType>();
  if (!input_indices_type || !input_shape_type || !new_shape_type) {
    return failure();
  }

  if (input_indices_type.getRank() != 2 || input_shape_type.getRank() != 1) {
    return failure();
  }
  Location loc = this->getLoc();

  Value num_values = builder.create<tensor::DimOp>(loc, operands[0], 0);
  Value new_rank = builder.create<tensor::DimOp>(loc, operands[2], 0);
  // output indices
  SmallVector<Value, 2> output_indices_shape_values;
  output_indices_shape_values.push_back(num_values);
  output_indices_shape_values.push_back(new_rank);
  Value output_indices_shape =
      builder.create<tensor::FromElementsOp>(loc, output_indices_shape_values);
  reifiedReturnShapes.push_back(output_indices_shape);

  // output shape
  SmallVector<Value, 1> output_shape_shape_values;
  Value output_rank =
      builder.create<tensor::DimOp>(loc, operands[operands.size() - 1], 0);
  output_shape_shape_values.push_back(output_rank);
  Value output_shape_shape =
      builder.create<tensor::FromElementsOp>(loc, output_shape_shape_values);
  reifiedReturnShapes.push_back(output_shape_shape);
  return success();
}

LogicalResult SparseReshapeOp::verify() {
  auto input_indices_type =
      this->getInputIndices().getType().template dyn_cast<RankedTensorType>();
  auto input_shape_type =
      this->getInputShape().getType().template dyn_cast<RankedTensorType>();
  auto new_shape_type =
      this->getNewShape().getType().template dyn_cast<RankedTensorType>();
  auto output_indices_type =
      this->getOutputIndices().getType().template dyn_cast<RankedTensorType>();
  auto output_shape_type =
      this->getOutputShape().getType().template dyn_cast<RankedTensorType>();

  if (!input_indices_type || !input_shape_type || !new_shape_type) {
    return this->emitOpError() << "only support ranked input.\n";
  }
  if (!input_indices_type.getElementType().isInteger(64) ||
      !input_shape_type.getElementType().isInteger(64) ||
      !new_shape_type.getElementType().isInteger(64)) {
    return this->emitOpError() << "only support int64 input.\n";
  }
  if (input_indices_type.getRank() != 2 || output_indices_type.getRank() != 2) {
    return this->emitOpError() << "Input/Output indices must be a matrix.\n";
  }
  if (input_shape_type.getRank() != 1 || new_shape_type.getRank() != 1 ||
      output_shape_type.getRank() != 1) {
    return this->emitOpError()
           << "Input/Output shape and new shape must be a vector.\n";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SparseFillEmptyRowsOp
//===----------------------------------------------------------------------===//

LogicalResult SparseFillEmptyRowsOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  SparseFillEmptyRowsOp::Adaptor adaptor(operands);
  // index 0
  auto indices_type = adaptor.getIndices().getType().cast<RankedTensorType>();
  // index 1
  auto values_type = adaptor.getValues().getType().cast<RankedTensorType>();
  // index 2
  auto dense_shape_type =
      adaptor.getDenseShape().getType().cast<RankedTensorType>();
  // index 3
  auto default_value_type =
      adaptor.getDefaultValue().getType().cast<RankedTensorType>();

  Location loc = this->getLoc();

  Value num_indices = builder.create<tensor::DimOp>(loc, operands[0], 0);
  Value rank = builder.create<tensor::DimOp>(loc, operands[0], 1);
  Value idx_zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  // TODO(lanbo.llb): Handle corner case when dense_rows == 0
  Value dense_rows = builder.create<arith::IndexCastOp>(
      loc, builder.getIndexType(),
      builder.create<tensor::ExtractOp>(loc, operands[2], idx_zero));
  // outputs:
  // 0: output_indices
  // 1: output_values
  // 2: empty_row_indicator
  // 3: reverse_index_map
  // 4: output_elements(scalar)

  // However N_full can only be determined by the values from indices
  // so we allocate a output_indices/output_values to max possible size, which
  // is `N + dense_rows`
  Value N_full = builder.create<arith::AddIOp>(loc, num_indices, dense_rows);

  // output indices: {N + dense_rows, rank}
  SmallVector<Value, 2> output_indices_shape_values;
  output_indices_shape_values.push_back(N_full);
  output_indices_shape_values.push_back(rank);
  Value output_indices_shape =
      builder.create<tensor::FromElementsOp>(loc, output_indices_shape_values);
  reifiedReturnShapes.push_back(output_indices_shape);

  // output values: {N + dense_rows}
  SmallVector<Value, 2> output_values_shape_values;
  output_values_shape_values.push_back(N_full);
  Value output_values_shape =
      builder.create<tensor::FromElementsOp>(loc, output_values_shape_values);
  reifiedReturnShapes.push_back(output_values_shape);

  // empty indicator: {dense_rows}
  SmallVector<Value, 1> empty_row_indicator_shape_values;
  empty_row_indicator_shape_values.push_back(dense_rows);
  Value empty_row_indicator_shape = builder.create<tensor::FromElementsOp>(
      loc, empty_row_indicator_shape_values);
  reifiedReturnShapes.push_back(empty_row_indicator_shape);

  // reverse index map: {N}
  SmallVector<Value, 1> reverse_index_map_shape_values;
  reverse_index_map_shape_values.push_back(num_indices);
  Value reverse_index_map_shape = builder.create<tensor::FromElementsOp>(
      loc, reverse_index_map_shape_values);
  reifiedReturnShapes.push_back(reverse_index_map_shape);

  // output_elements
  Value idx_one = builder.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Value, 1> output_elements_shape_values;
  output_elements_shape_values.push_back(idx_one);
  Value output_elements_shape =
      builder.create<tensor::FromElementsOp>(loc, output_elements_shape_values);
  reifiedReturnShapes.push_back(output_elements_shape);

  return success();
}

LogicalResult SparseFillEmptyRowsOp::verify() {
  auto indices_type = this->getIndices().getType().dyn_cast<RankedTensorType>();
  auto values_type = this->getValues().getType().dyn_cast<RankedTensorType>();
  auto dense_shape_type =
      this->getDenseShape().getType().dyn_cast<RankedTensorType>();
  auto default_value_type =
      this->getDefaultValue().getType().dyn_cast<RankedTensorType>();

  if (!indices_type || !values_type || !default_value_type) {
    return failure();
  }

  if (indices_type.getRank() != 2) {
    return this->emitOpError() << "indices must be a matrix";
  }
  if (values_type.getRank() != 1) {
    return this->emitOpError() << "values must be a vector";
  }
  if (default_value_type.getRank() != 0) {
    return this->emitOpError() << "default_value must be a scalar";
  }
  if (!dense_shape_type.hasStaticShape()) {
    return this->emitOpError() << "DISC only support static-rank optimization, "
                                  "thus dense_shape should has static shape";
  }

  if (dense_shape_type.getRank() != 1) {
    return this->emitOpError() << "dense_shape must be a vector";
  }

  auto output_indices_type =
      this->getOutputIndices().getType().dyn_cast<RankedTensorType>();
  auto output_values_type =
      this->getOutputValues().getType().dyn_cast<RankedTensorType>();
  auto empty_row_indicator_type =
      this->getEmptyRowIndicator().getType().dyn_cast<RankedTensorType>();
  auto reverse_index_map_type =
      this->getReverseIndexMap().getType().dyn_cast<RankedTensorType>();
  auto output_elements_type =
      this->getOutputElements().getType().dyn_cast<RankedTensorType>();
  if (!output_indices_type || !output_values_type ||
      !empty_row_indicator_type || !reverse_index_map_type ||
      !output_elements_type) {
    return failure();
  }

  if (output_indices_type.getRank() != 2) {
    return this->emitOpError() << "output_indices must be a matrix";
  }

  if (output_values_type.getRank() != 1 ||
      empty_row_indicator_type.getRank() != 1 ||
      reverse_index_map_type.getRank() != 1 ||
      output_elements_type.getRank() != 1) {
    return this->emitOpError()
           << "outputs 1-4 for mhlo_disc::sparse_fill_empty_rows must be a "
              "vector";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SparseSegmentReductionOp
//===----------------------------------------------------------------------===//

LogicalResult SparseSegmentReductionOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  SparseSegmentReductionOp::Adaptor adaptor(operands);
  Location loc = this->getLoc();
  auto data_type = adaptor.getData().getType().cast<RankedTensorType>();
  auto indices_type = adaptor.getIndices().getType().cast<RankedTensorType>();
  auto segment_ids_type =
      adaptor.getSegmentIds().getType().cast<RankedTensorType>();
  auto input_rank = data_type.getRank();
  SmallVector<Value, 2> output_shape_values;
  Value segment_size = builder.create<tensor::DimOp>(loc, operands[2], 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value last_segment_index =
      builder.create<arith::SubIOp>(loc, segment_size, one);
  Value last_segment_id_plus_one = builder.create<arith::AddIOp>(
      loc,
      builder.create<arith::IndexCastOp>(
          loc, builder.getIndexType(),
          builder.create<tensor::ExtractOp>(loc, operands[2],
                                            last_segment_index)),
      one);
  output_shape_values.push_back(last_segment_id_plus_one);
  for (auto i = 1; i < input_rank; i++) {
    output_shape_values.push_back(
        builder.create<tensor::DimOp>(loc, operands[0], i));
  }
  Value output_shape =
      builder.create<tensor::FromElementsOp>(loc, output_shape_values);
  reifiedReturnShapes.push_back(output_shape);
  return success();
}

LogicalResult SparseSegmentReductionOp::verify() {
  auto data_type = this->getData().getType().dyn_cast<RankedTensorType>();
  auto indices_type = this->getIndices().getType().dyn_cast<RankedTensorType>();
  auto segment_ids_type =
      this->getSegmentIds().getType().dyn_cast<RankedTensorType>();

  if (!data_type || !indices_type || !segment_ids_type) {
    return failure();
  }

  if (indices_type.getRank() != 1) {
    return this->emitOpError() << "indices should be a vector";
  }
  if (segment_ids_type.getRank() != 1) {
    return this->emitOpError() << "segment_ids should be a vector";
  }
  // only check when has static shape
  if (indices_type.hasStaticShape() && segment_ids_type.hasStaticShape()) {
    auto num_indices = indices_type.getDimSize(0);
    auto num_segment_ids = segment_ids_type.getDimSize(0);
    if (num_indices != num_segment_ids) {
      return this->emitOpError()
             << "segment_ids and indices should have same size";
    }
  }

  if (data_type.getRank() < 1) {
    return this->emitOpError() << "input must be at least rank 1";
  }

  auto output_type = this->getOutput().getType().dyn_cast<RankedTensorType>();
  if (!output_type) {
    return failure();
  }

  if (output_type.getRank() != data_type.getRank()) {
    return this->emitOpError() << "output must have the same rank as input";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SparseSegmentReductionWithEmptyRowsOp
//===----------------------------------------------------------------------===//

LogicalResult SparseSegmentReductionWithEmptyRowsOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  SparseSegmentReductionWithEmptyRowsOp::Adaptor adaptor(operands);
  Location loc = this->getLoc();
  auto data_type = adaptor.getData().getType().cast<RankedTensorType>();
  auto indices_type = adaptor.getIndices().getType().cast<RankedTensorType>();
  auto input_rank = data_type.getRank();
  SmallVector<Value, 2> output_shape_values;
  SmallVector<Value, 2> empty_rows_indicator_shape_values;
  Value idx_zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  // dense shape should be rank 2 here, other wise we should multiply
  // dim 0 ~ rank-1.
  Value dense_rows = builder.create<arith::IndexCastOp>(
      loc, builder.getIndexType(),
      builder.create<tensor::ExtractOp>(loc, operands[3], idx_zero));
  output_shape_values.push_back(dense_rows);

  for (auto i = 1; i < input_rank; i++) {
    output_shape_values.push_back(
        builder.create<tensor::DimOp>(loc, operands[0], i));
  }
  Value output_shape =
      builder.create<tensor::FromElementsOp>(loc, output_shape_values);
  reifiedReturnShapes.push_back(output_shape);

  empty_rows_indicator_shape_values.push_back(dense_rows);
  Value empty_rows_indicator_shape =
      builder.create<tensor::FromElementsOp>(loc, output_shape_values);
  reifiedReturnShapes.push_back(empty_rows_indicator_shape);
  return success();
}

LogicalResult SparseSegmentReductionWithEmptyRowsOp::verify() {
  auto data_type = this->getData().getType().dyn_cast<RankedTensorType>();
  auto indices_type = this->getIndices().getType().dyn_cast<RankedTensorType>();
  auto segment_ids_type =
      this->getUnfilledSegmentIds().getType().dyn_cast<RankedTensorType>();

  if (!data_type || !indices_type || !segment_ids_type) {
    return failure();
  }

  if (indices_type.getRank() != 2) {
    return this->emitOpError() << "indices should be a matrix";
  }
  if (segment_ids_type.getRank() != 1) {
    return this->emitOpError() << "unfilled_segment_ids should be a vector";
  }

  // only support for rank 2, now
  // TODO(lanbo.llb): support other rank
  if (data_type.getRank() != 2) {
    return this->emitOpError()
           << "input must be matrix since other rank is not supported";
  }

  auto output_type = this->getOutput().getType().dyn_cast<RankedTensorType>();
  if (!output_type) {
    return failure();
  }

  if (output_type.getRank() != data_type.getRank()) {
    return this->emitOpError() << "output must have the same rank as input";
  }

  auto indicator_type =
      this->getEmptyRowIndicator().getType().dyn_cast<RankedTensorType>();
  if (!indicator_type) {
    return failure();
  }

  if (indicator_type.getRank() != 1) {
    return this->emitOpError() << "empty_row_indicator must be a vector";
  }

  return success();
}

// CustomCallV2Op
//===----------------------------------------------------------------------===//

LogicalResult CustomCallV2Op::verify() {
  SmallVector<std::string> inputLayouts = parseInputLayouts();
  SmallVector<std::string> expectedInputLayouts = parseExpectedInputLayouts();
  if (inputLayouts.size() != expectedInputLayouts.size())
    return this->emitOpError() << "mismatch number of layouts for "
                                  "input_layouts and expected_input_layouts\n";

  if (inputLayouts.size() != 0 && inputLayouts.size() != this->getNumOperands())
    return this->emitOpError()
           << "mismatch number of input layouts and number of inputs\n";

  SmallVector<std::string> outputLayouts = parseOutputLayouts();
  SmallVector<std::string> expectedOutputLayouts = parseExpectedOutputLayouts();
  if (outputLayouts.size() != expectedOutputLayouts.size())
    return this->emitOpError()
           << "mismatch number of layouts for output_layouts and "
              "expected_output_layouts\n";
  if (outputLayouts.size() != 0 &&
      outputLayouts.size() != this->getNumResults())
    return this->emitOpError()
           << "mismatch number of output layouts and number of outputs\n";

  auto checkLayouts = [&](SmallVector<std::string>& lhs,
                          SmallVector<std::string>& rhs, TypeRange typeRange) {
    for (const auto& z : llvm::zip(lhs, rhs, typeRange)) {
      if (std::get<0>(z).size() != std::get<1>(z).size()) return failure();

      auto ty = std::get<2>(z).dyn_cast<RankedTensorType>();
      if (ty && ty.getRank() != std::get<0>(z).size()) return failure();
    }
    return success();
  };
  if (failed(checkLayouts(inputLayouts, expectedInputLayouts,
                          this->getOperandTypes())))
    return this->emitOpError()
           << "mismatch input layout or expected input layout setting";
  if (failed(checkLayouts(outputLayouts, expectedOutputLayouts,
                          this->getResultTypes())))
    return this->emitOpError()
           << "mismatch output layout or expected output layout setting";

  return success();
}

// WhereOp
//===----------------------------------------------------------------------===//

LogicalResult WhereOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  WhereOp::Adaptor adaptor(operands);
  Location loc = this->getLoc();
  auto input_type = adaptor.getInput().getType().cast<RankedTensorType>();

  SmallVector<Value, 2> index_shape_values, num_output_elements_shape_values;

  Value idx_one = builder.create<arith::ConstantIndexOp>(loc, 1);
  auto input_rank = input_type.getRank();
  Value num_input_elements = idx_one;
  for (int i = 0; i < input_rank; i++) {
    Value dim_i = builder.create<tensor::DimOp>(loc, operands[0], i);
    num_input_elements =
        builder.create<arith::MulIOp>(loc, num_input_elements, dim_i);
  }
  Value input_rank_value =
      builder.create<arith::ConstantIndexOp>(loc, input_type.getRank());
  index_shape_values.push_back(num_input_elements);
  index_shape_values.push_back(input_rank_value);
  Value index_shape =
      builder.create<tensor::FromElementsOp>(loc, index_shape_values);
  reifiedReturnShapes.push_back(index_shape);

  num_output_elements_shape_values.push_back(idx_one);
  Value num_output_elements_shape = builder.create<tensor::FromElementsOp>(
      loc, num_output_elements_shape_values);
  reifiedReturnShapes.push_back(num_output_elements_shape);

  return success();
}

LogicalResult WhereOp::verify() {
  auto input_type = this->getInput().getType().dyn_cast<RankedTensorType>();

  if (!input_type) {
    return failure();
  }

  auto index_type = this->getIndex().getType().dyn_cast<RankedTensorType>();
  if (!index_type) {
    return failure();
  }

  if (index_type.getRank() != 2) {
    return this->emitOpError() << "output must be a matrix";
  }

  return success();
}

}  // namespace mhlo_disc
}  // namespace mlir

#define GET_OP_CLASSES
#include "mlir/disc/IR/hlo_disc_ops.cc.inc"

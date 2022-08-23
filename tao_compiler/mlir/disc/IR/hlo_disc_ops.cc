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

#include "tensorflow/compiler/mlir/disc/IR/hlo_disc_ops.h"

#include <mutex>
#include <unordered_map>

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "tensorflow/compiler/mlir/disc/IR/custom_call_base.h"

namespace mlir {
namespace mhlo_disc {

using llvm::StringRef;

template <typename T>
static LogicalResult Verify(T op) {
  return success();
}

//===----------------------------------------------------------------------===//
// mhlo disc Dialect Constructor
//===----------------------------------------------------------------------===//

MhloDiscDialect::MhloDiscDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context, TypeID::get<MhloDiscDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/disc/IR/hlo_disc_ops.cc.inc"

      >();
  context->loadDialect<tensor::TensorDialect>();
}

//===----------------------------------------------------------------------===//
// H2DOp
//===----------------------------------------------------------------------===//

LogicalResult H2DOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  return mhlo::deriveShapeFromOperand(&builder, getOperation(), operands[0],
                                      &reifiedReturnShapes);
}

LogicalResult H2DOp::verify() { return Verify(*this); }

//===----------------------------------------------------------------------===//
// D2HOp
//===----------------------------------------------------------------------===//

LogicalResult D2HOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  return mhlo::deriveShapeFromOperand(&builder, getOperation(), operands[0],
                                      &reifiedReturnShapes);
}

LogicalResult D2HOp::verify() { return Verify(*this); }

//===----------------------------------------------------------------------===//
// CustomCallOp
//===----------------------------------------------------------------------===//

LogicalResult CustomCallOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  CustomCallOp::Adaptor adaptor(operands);
  ValueRange args = adaptor.args();
  StringRef target = call_target_name();
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
  auto inputTy = op->input().getType().template dyn_cast<RankedTensorType>();
  auto scaleTy = op->scale().getType().template dyn_cast<RankedTensorType>();
  auto zeroPointTy =
      op->zero_point().getType().template dyn_cast<RankedTensorType>();
  auto resultTy = op->result().getType().template dyn_cast<RankedTensorType>();
  if (!inputTy || !scaleTy || !zeroPointTy || !resultTy)
    return op->emitOpError() << "only support ranked input.\n";
  if (inputTy.getShape() != resultTy.getShape())
    return op->emitOpError() << "input and result have mismatch shape.\n";
  if (scaleTy.getRank() != zeroPointTy.getRank())
    return op->emitOpError() << "scale and zero_point have mismatch rank.\n";
  auto axis = op->axis().template getValues<int64_t>();
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

//===----------------------------------------------------------------------===//
// DequantizeOp
//===----------------------------------------------------------------------===//

LogicalResult DequantizeOp::verify() { return QuantVerify(this); }

//===----------------------------------------------------------------------===//
// QuantizedDotGeneralOp
//===----------------------------------------------------------------------===//

template <typename T>
LogicalResult CommonVerifyForQuantizedComputeIntensiveOp(T* op) {
  auto inputTy = op->input().getType().template dyn_cast<RankedTensorType>();
  auto weightTy = op->weight().getType().template dyn_cast<RankedTensorType>();
  auto resultTy = op->result().getType().template dyn_cast<RankedTensorType>();

  if (!inputTy || !weightTy || !resultTy ||
      inputTy.getRank() != weightTy.getRank() ||
      inputTy.getRank() != resultTy.getRank()) {
    return op->emitOpError()
           << "input, weight and result should have the same rank.\n";
  }

  auto inputScaleTy =
      op->input_scale().getType().template dyn_cast<RankedTensorType>();
  auto inputZeroPointTy =
      op->input_zero_point().getType().template dyn_cast<RankedTensorType>();
  if (!inputScaleTy || !inputZeroPointTy || inputScaleTy.getRank() != 0 ||
      inputZeroPointTy.getRank() != 0) {
    return op->emitOpError() << "input_scale and input_zero_point only support "
                                "per-tensor quantization\n";
  }

  auto resultScaleTy =
      op->result_scale().getType().template dyn_cast<RankedTensorType>();
  auto resultZeroPointTy =
      op->result_zero_point().getType().template dyn_cast<RankedTensorType>();
  if (!resultScaleTy || !resultZeroPointTy || resultScaleTy.getRank() != 0 ||
      resultZeroPointTy.getRank() != 0) {
    return op->emitOpError() << "result_scale and result_zero_point only "
                                "support per-tensor quantization\n";
  }

  auto weightScaleTy =
      op->weight_scale().getType().template dyn_cast<RankedTensorType>();
  auto weightZeroPointTy =
      op->weight_zero_point().getType().template dyn_cast<RankedTensorType>();
  if (!weightScaleTy || !weightZeroPointTy ||
      weightScaleTy.getShape() != weightZeroPointTy.getShape()) {
    return op->emitOpError()
           << "weight_scale and weight_zero_point have mismatch shape\n";
  }
  auto axis = op->axis().template getValues<int64_t>();
  if (axis.size() != weightScaleTy.getRank())
    return op->emitOpError() << "num of quantized axes (len(axis)) is not "
                                "equal to the rank of weight_scale tensor\n";
  return success();
}

LogicalResult QuantizedDotGeneralOp::verify() {
  return CommonVerifyForQuantizedComputeIntensiveOp(this);
}

//===----------------------------------------------------------------------===//
// QuantizedDynamicConvOp
//===----------------------------------------------------------------------===//

LogicalResult QuantizedDynamicConvOp::verify() {
  return CommonVerifyForQuantizedComputeIntensiveOp(this);
}

}  // namespace mhlo_disc
}  // namespace mlir

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/disc/IR/hlo_disc_ops.cc.inc"

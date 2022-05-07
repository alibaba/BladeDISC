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

}  // namespace mhlo_disc
}  // namespace mlir

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/disc/IR/hlo_disc_ops.cc.inc"

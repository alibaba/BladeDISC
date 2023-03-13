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

#include "mlir/disc/transforms/disc_to_llvm_common.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace disc_ral {

LogicalResult RemoveUselessUnrealizedConversionCastOp::matchAndRewrite(
    UnrealizedConversionCastOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto in_type = op.getInputs().getTypes().front();
  auto out_type = op.getOutputs().getTypes().front();

  bool isSignToSignless =
      (in_type.isSignedInteger() || in_type.isUnsignedInteger()) &&
      (out_type.isSignlessInteger()) &&
      (in_type.getIntOrFloatBitWidth() == out_type.getIntOrFloatBitWidth());
  bool isSignlessToSign =
      (out_type.isSignedInteger() || out_type.isUnsignedInteger()) &&
      (in_type.isSignlessInteger()) &&
      (in_type.getIntOrFloatBitWidth() == out_type.getIntOrFloatBitWidth());

  // this pattern only used for uint/sint->int or int -> uint/sint conversion
  // now since it has no semantic in llvm Dialect
  if (isSignToSignless || isSignlessToSign) {
    op.replaceAllUsesWith(adaptor.getOperands());
    op.erase();
    return success();
  }
  return op.emitOpError() << "unexpected unrealized conversion cast op";
}

}  // namespace disc_ral
}  // namespace mlir

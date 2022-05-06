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

#ifndef MLIR_DISC_TRANSFORMS_DISC_DISC_TO_LLVM_COMMON_H_
#define MLIR_DISC_TRANSFORMS_DISC_DISC_TO_LLVM_COMMON_H_

// This file implements soem common patterns for lowering DISC to LLVM

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {

struct LogicalResult;

namespace disc_ral {

struct RemoveUselessUnrealizedConversionCastOp
    : public ConvertOpToLLVMPattern<UnrealizedConversionCastOp> {
  using ConvertOpToLLVMPattern<
      UnrealizedConversionCastOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      UnrealizedConversionCastOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override;
};

}  // namespace disc_ral
}  // namespace mlir

#endif  // MLIR_DISC_TRANSFORMS_DISC_DISC_TO_LLVM_COMMON_H_

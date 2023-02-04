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

#ifndef MLIR_DISC_TRANSFORMS_DISC_DISC_LOWER_TO_GPU_OPS_COMMON_H_
#define MLIR_DISC_TRANSFORMS_DISC_DISC_LOWER_TO_GPU_OPS_COMMON_H_

// This file implements common patterns for GPUToNVVM and GPUToROCDL

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/disc/transforms/disc_to_llvm_common.h"

namespace mlir {

struct LogicalResult;

namespace disc_ral {

// Common base for load and store operations on MemRefs.  Restricts the match
// to supported MemRef types. Provides functionality to emit code accessing a
// specific element of the underlying data buffer.
template <typename Derived>
struct LoadStoreOpLowering : public ConvertOpToLLVMPattern<Derived> {
  using ConvertOpToLLVMPattern<Derived>::ConvertOpToLLVMPattern;
  using ConvertOpToLLVMPattern<Derived>::isConvertibleAndHasIdentityMaps;
  using Base = LoadStoreOpLowering<Derived>;

  LogicalResult match(Derived op) const override {
    MemRefType type = op.getMemRefType();
    return isConvertibleAndHasIdentityMaps(type) ? success() : failure();
  }
};

struct GenericAtomicRMWOpLoweringWithBitcast
    : public LoadStoreOpLowering<memref::GenericAtomicRMWOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(
      memref::GenericAtomicRMWOp atomicOp, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override;
};

}  // namespace disc_ral
}  // namespace mlir

#endif  // MLIR_DISC_TRANSFORMS_DISC_DISC_LOWER_TO_GPU_OPS_COMMON_H_

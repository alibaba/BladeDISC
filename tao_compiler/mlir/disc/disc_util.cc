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
#include "tensorflow/compiler/mlir/disc/disc_util.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "tensorflow/compiler/mlir/disc/transforms/placement_utils.h"

namespace mlir {
namespace disc_ral {

bool IsSmallBuffer(Value alloc) {
  constexpr unsigned kMaximumSizeInBytes = 128;
  constexpr unsigned kBitwidthOfIndexType = 64;

  auto type = alloc.getType().dyn_cast<ShapedType>();
  if (!type || !type.hasStaticShape()) return false;

  // For index types, use the provided size, as the type does not know.
  unsigned int bitwidth = type.getElementType().isIndex()
                              ? kBitwidthOfIndexType
                              : type.getElementTypeBitWidth();
  return type.getNumElements() * bitwidth <= kMaximumSizeInBytes * 8;
}

bool IsSmallCpuBuffer(Value alloc) {
  if (placement_utils::isGpuMemRef(alloc)) return false;
  return IsSmallBuffer(alloc);
}

bool IsSmallCpuAlloc(Value alloc) {
  return IsSmallCpuBuffer(alloc) && alloc.getDefiningOp<memref::AllocOp>();
}

bool IsOpWriteValue(Operation* op, Value value) {
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 2> effects;
  MemoryEffectOpInterface interface = dyn_cast<MemoryEffectOpInterface>(op);
  // Suppose that value without `MemoryEffectOpInterface` is readonly.
  if (!interface) return false;

  interface.getEffectsOnValue(value, effects);
  return llvm::any_of(
      effects, [](const mlir::MemoryEffects::EffectInstance& instance) {
        return mlir::isa<mlir::MemoryEffects::Write>(instance.getEffect());
      });
}

bool IsMemRefAliasOp(Operation* op) {
  return dyn_cast<ViewLikeOpInterface>(op) != nullptr;
}

Value getRootMemRef(Value memref) {
  Value rootMemRef = memref;
  while (Operation* operandOp = rootMemRef.getDefiningOp()) {
    if (!isa<memref::SubViewOp, memref::ViewOp, memref::CastOp,
             memref::ReinterpretCastOp>(operandOp))
      break;
    rootMemRef = operandOp->getOperand(0);
  }
  return rootMemRef;
}

bool isSameUnderlineBuffer(Value lhs, Value rhs) {
  return getRootMemRef(lhs) == getRootMemRef(rhs);
}

}  // namespace disc_ral
}  // namespace mlir
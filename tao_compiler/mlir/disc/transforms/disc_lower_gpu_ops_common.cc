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

#include "tensorflow/compiler/mlir/disc/transforms/disc_lower_gpu_ops_common.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace disc_ral {

LogicalResult RemoveUselessUnrealizedConversionCastOp::matchAndRewrite(
    UnrealizedConversionCastOp op, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto in_type = op.inputs().getTypes().front();
  auto out_type = op.outputs().getTypes().front();
  // this pattern only used for uint/sint->int conversion now
  // since it has no semantic in llvm Dialect
  if ((in_type.isSignedInteger() || in_type.isUnsignedInteger()) &&
      (out_type.isSignlessInteger()) &&
      in_type.getIntOrFloatBitWidth() == out_type.getIntOrFloatBitWidth()) {
    op.replaceAllUsesWith(adaptor.getOperands());
    op.erase();
    return success();
  }
  return op.emitOpError() << "unexpected unrealized conversion cast op";
}

/// Wrap a llvm.cmpxchg operation in a while loop so that the operation can be
/// retried until it succeeds in atomically storing a new value into memory.
///
/// Currently we only deal with 32/64-bit types. Non-integer type will be cast
/// to the corresponding integer type. Note that `llvm.cmpxchg` only supports
/// integet or pointer types, and NVIDIA only supports atomicCAS on 32 bit and
/// 64 bit integers.
///
///      +---------------------------------+
///      |   <code before the AtomicRMWOp> |
///      |   <compute initial %loaded>     |
///      |   br loop(%loaded)              |
///      +---------------------------------+
///             |
///  -------|   |
///  |      v   v
///  |   +--------------------------------+
///  |   | loop(%loaded):                 |
///  |   |   <body contents>              |
///  |   |   %pair = cmpxchg              |
///  |   |   %ok = %pair[0]               |
///  |   |   %new = %pair[1]              |
///  |   |   cond_br %ok, end, loop(%new) |
///  |   +--------------------------------+
///  |          |        |
///  |-----------        |
///                      v
///      +--------------------------------+
///      | end:                           |
///      |   <code after the AtomicRMWOp> |
///      +--------------------------------+
///
LogicalResult GenericAtomicRMWOpLoweringWithBitcast::matchAndRewrite(
    memref::GenericAtomicRMWOp atomicOp, OpAdaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  Location loc = atomicOp.getLoc();
  Type valueType = typeConverter->convertType(atomicOp.getResult().getType());

  // Split the block into initial, loop, and ending parts.
  Block* initBlock = rewriter.getInsertionBlock();
  Block* endBlock =
      rewriter.splitBlock(initBlock, std::next(Block::iterator(atomicOp)));
  Block* loopBlock =
      rewriter.createBlock(initBlock->getParent(),
                           std::next(Region::iterator(initBlock)), valueType);

  // Compute the loaded value and branch to the loop block.
  rewriter.setInsertionPointToEnd(initBlock);
  auto memRefType = atomicOp.memref().getType().cast<MemRefType>();
  auto dataPtr = getStridedElementPtr(loc, memRefType, adaptor.memref(),
                                      adaptor.indices(), rewriter);
  Value init = rewriter.create<LLVM::LoadOp>(loc, dataPtr);
  rewriter.create<LLVM::BrOp>(loc, init, loopBlock);

  // Prepare the body of the loop block.
  rewriter.setInsertionPointToStart(loopBlock);

  // Clone the memref::GenericAtomicRMWOp region and extract the result.
  auto loopArgument = loopBlock->getArgument(0);
  BlockAndValueMapping mapping;
  mapping.map(atomicOp.getCurrentValue(), loopArgument);
  Block& entryBlock = atomicOp.atomic_body().front();
  for (auto& nestedOp : entryBlock.without_terminator()) {
    Operation* clone = rewriter.clone(nestedOp, mapping);
    mapping.map(nestedOp.getResults(), clone->getResults());
  }
  Value result = mapping.lookup(entryBlock.getTerminator()->getOperand(0));

  // Prepare the epilog of the loop block.
  // Append the cmpxchg op to the end of the loop block.
  auto successOrdering = LLVM::AtomicOrdering::acq_rel;
  auto failureOrdering = LLVM::AtomicOrdering::monotonic;
  auto boolType = IntegerType::get(rewriter.getContext(), 1);
  auto pairType = LLVM::LLVMStructType::getLiteral(rewriter.getContext(),
                                                   {valueType, boolType});

  // Cast datatype for non integer 32/64 type.
  Type mayCastedType = valueType;
  Value mayCastedDataPtr = dataPtr;
  LLVM::LLVMStructType mayCastedPairType = pairType;
  Value mayCastedLoopArgument = loopArgument;
  Value mayCastedResult = result;
  if (valueType.isF32()) {
    mayCastedType = rewriter.getI32Type();
    mayCastedDataPtr = rewriter.create<LLVM::BitcastOp>(
        loc, LLVM::LLVMPointerType::get(mayCastedType), dataPtr);
    mayCastedPairType = LLVM::LLVMStructType::getLiteral(
        rewriter.getContext(), {mayCastedType, boolType});
    mayCastedLoopArgument =
        rewriter.create<LLVM::BitcastOp>(loc, mayCastedType, loopArgument);
    mayCastedResult =
        rewriter.create<LLVM::BitcastOp>(loc, mayCastedType, result);
  } else if (valueType.isF64()) {
    mayCastedType = rewriter.getI64Type();
    mayCastedDataPtr = rewriter.create<LLVM::BitcastOp>(
        loc, LLVM::LLVMPointerType::get(mayCastedType), dataPtr);
    mayCastedPairType = LLVM::LLVMStructType::getLiteral(
        rewriter.getContext(), {mayCastedType, boolType});
    mayCastedLoopArgument =
        rewriter.create<LLVM::BitcastOp>(loc, mayCastedType, loopArgument);
    mayCastedResult =
        rewriter.create<LLVM::BitcastOp>(loc, mayCastedType, result);
  } else if (!valueType.isInteger(32) && !valueType.isInteger(64)) {
    // TODO: deal with bitwith < 32 bits. The basic idea is to load 32-bit
    // area and update sub-region according to the target bitwith, and finally
    // write back.
    return failure();
  }

  auto cmpxchg = rewriter.create<LLVM::AtomicCmpXchgOp>(
      loc, mayCastedPairType, mayCastedDataPtr, mayCastedLoopArgument,
      mayCastedResult, successOrdering, failureOrdering);
  // Extract the %new_loaded and %ok values from the pair.
  Value newLoaded = rewriter.create<LLVM::ExtractValueOp>(
      loc, mayCastedType, cmpxchg, rewriter.getI64ArrayAttr({0}));
  if (valueType != mayCastedType) {
    newLoaded = rewriter.create<LLVM::BitcastOp>(loc, valueType, newLoaded);
  }
  Value ok = rewriter.create<LLVM::ExtractValueOp>(
      loc, boolType, cmpxchg, rewriter.getI64ArrayAttr({1}));

  // Conditionally branch to the end or back to the loop depending on %ok.
  rewriter.create<LLVM::CondBrOp>(loc, ok, endBlock, ArrayRef<Value>(),
                                  loopBlock, newLoaded);

  // The 'result' of the atomic_rmw op is the newly loaded value.
  mapping.map(atomicOp.getResult(), newLoaded);
  rewriter.replaceOp(atomicOp, {newLoaded});

  return success();
}

}  // namespace disc_ral
}  // namespace mlir

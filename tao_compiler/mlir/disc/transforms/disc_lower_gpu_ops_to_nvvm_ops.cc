// Copyright 2021 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- LowerGpuOpsToNVVMOps.cpp - MLIR GPU to NVVM lowering passes --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to generate NVVMIR operations for higher-level
// GPU operations.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/FormatVariadic.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/NVGPUToNVVM/NVGPUToNVVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/disc_lower_gpu_ops_common.h"
#include "mlir/disc/transforms/placement_utils.h"
#include "mlir/lib/Conversion/GPUCommon/GPUOpsLowering.h"
#include "mlir/lib/Conversion/GPUCommon/IndexIntrinsicsOpLowering.h"
#include "mlir/lib/Conversion/GPUCommon/OpToFuncCallLowering.h"

namespace mlir {
namespace disc_ral {

namespace {

/// Import the GPU Ops to NVVM Patterns.
#include "GPUToNVVM.cpp.inc"

/// Conversion vector.store with align attribute to llvm.store
class VectorStoreWithAlignToLLVMPattern
    : public ConvertOpToLLVMPattern<vector::StoreOp> {
  using ConvertOpToLLVMPattern<vector::StoreOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      vector::StoreOp storeOp, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    // Only 1-D vectors can be lowered to LLVM.
    VectorType vectorTy = storeOp.getVectorType();
    if (vectorTy.getRank() > 1) return failure();
    auto alignAttr = storeOp->getAttrOfType<IntegerAttr>("alignment");
    if (!alignAttr) return failure();
    unsigned align = alignAttr.getInt();

    auto loc = storeOp->getLoc();
    MemRefType memRefTy = storeOp.getMemRefType();

    // Resolve address.
    auto vtype = cast<VectorType>(
        this->typeConverter->convertType(storeOp.getVectorType()));
    Value dataPtr = this->getStridedElementPtr(loc, memRefTy, adaptor.getBase(),
                                               adaptor.getIndices(), rewriter);
    // Casts a strided element pointer to a vector pointer.  The vector pointer
    // will be in the same address space as the incoming memref type.
    Value ptr;
    if ((*this->getTypeConverter()).useOpaquePointers()) {
      ptr = dataPtr;
    } else {
      unsigned addressSpace =
          *(*this->getTypeConverter()).getMemRefAddressSpace(memRefTy);
      auto pType = LLVM::LLVMPointerType::get(vtype, addressSpace);
      ptr = rewriter.create<LLVM::BitcastOp>(loc, pType, dataPtr);
    }

    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(
        storeOp, adaptor.getValueToStore(), ptr, align);
    return success();
  }
};

/// A pass that replaces all occurrences of GPU device operations with their
/// corresponding NVVM equivalent.
///
/// This pass only handles device code and is not meant to be run on GPU host
/// code.
struct DiscLowerGpuOpsToNVVMOpsPass
    : public DiscLowerGpuOpsToNVVMOpsPassBase<DiscLowerGpuOpsToNVVMOpsPass> {
  DiscLowerGpuOpsToNVVMOpsPass() = default;
  DiscLowerGpuOpsToNVVMOpsPass(unsigned indexBitwidth, bool hasRedux = false) {
    this->indexBitwidth = indexBitwidth;
  }

  void runOnOperation() override {
    gpu::GPUModuleOp m = getOperation();

    /// Customize the bitwidth used for the device side index computations.
    LowerToLLVMOptions options(
        m.getContext(),
        DataLayout(cast<DataLayoutOpInterface>(m.getOperation())));
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);

    // Apply in-dialect lowering. In-dialect lowering will replace
    // ops which need to be lowered further, which is not supported by a
    // single conversion pass.
    {
      RewritePatternSet patterns(m.getContext());
      populateGpuRewritePatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(m, std::move(patterns))))
        return signalPassFailure();
    }

    LLVMTypeConverter converter(m.getContext(), options);
    // NVVM uses alloca in the default address space to represent private
    // memory allocations, so drop private annotations. NVVM uses address
    // space 3 for shared memory. NVVM uses the default address space to
    // represent global memory.
    populateGpuMemorySpaceAttributeConversions(
        converter, [](gpu::AddressSpace space) -> unsigned {
          switch (space) {
            case gpu::AddressSpace::Global:
              return static_cast<unsigned>(
                  NVVM::NVVMMemorySpace::kGlobalMemorySpace);
            case gpu::AddressSpace::Workgroup:
              return static_cast<unsigned>(
                  NVVM::NVVMMemorySpace::kSharedMemorySpace);
            case gpu::AddressSpace::Private:
              return 0;
          }
          llvm_unreachable("unknown address space enum value");
          return 0;
        });

    // Lowering for MMAMatrixType.
    converter.addConversion([&](gpu::MMAMatrixType type) -> Type {
      return convertMMAToLLVMType(type);
    });
    RewritePatternSet llvmPatterns(m.getContext());

    // Add the fix before other conversions.
    llvmPatterns.add<GenericAtomicRMWOpLoweringWithBitcast>(
        converter, /* PatternBenefit */ 3);
    llvmPatterns.add<RemoveUselessUnrealizedConversionCastOp>(converter);
    llvmPatterns.add<VectorStoreWithAlignToLLVMPattern>(converter,
                                                        /* PatternBenefit */ 3);
    arith::populateArithToLLVMConversionPatterns(converter, llvmPatterns);
    cf::populateControlFlowToLLVMConversionPatterns(converter, llvmPatterns);
    populateVectorToLLVMConversionPatterns(converter, llvmPatterns);
    populateFuncToLLVMConversionPatterns(converter, llvmPatterns);
    populateFinalizeMemRefToLLVMConversionPatterns(converter, llvmPatterns);
    populateGpuToNVVMConversionPatterns(converter, llvmPatterns);
    populateNVGPUToNVVMConversionPatterns(converter, llvmPatterns);
    // Put the math conversioin after GpuToNVVM conversions as some math ops
    // are intended to be converted to nvvm intrinsics.
    populateMathToLLVMConversionPatterns(converter, llvmPatterns);
    memref::populateExpandStridedMetadataPatterns(llvmPatterns);
    populateGpuWMMAToNVVMConversionPatterns(converter, llvmPatterns);
    if (this->hasRedux)
      populateGpuSubgroupReduceOpLoweringPattern(converter, llvmPatterns);

    LLVMConversionTarget target(getContext());
    configureGpuToNVVMConversionLegality(target);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<arith::ArithDialect, math::MathDialect,
                             cf::ControlFlowDialect>();
    target.addIllegalOp<UnrealizedConversionCastOp>();
    if (failed(applyPartialConversion(m, target, std::move(llvmPatterns))))
      signalPassFailure();
  }
};

}  // anonymous namespace

std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
createDiscLowerGpuOpsToNVVMOpsPass(unsigned indexBitwidth,
                                   bool hasRedux = false) {
  return std::make_unique<DiscLowerGpuOpsToNVVMOpsPass>(indexBitwidth,
                                                        hasRedux);
}

}  // namespace disc_ral
}  // namespace mlir

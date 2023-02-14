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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/disc_lower_gpu_ops_common.h"
#include "mlir/lib/Conversion/GPUCommon/GPUOpsLowering.h"
#include "mlir/lib/Conversion/GPUCommon/IndexIntrinsicsOpLowering.h"
#include "mlir/lib/Conversion/GPUCommon/OpToFuncCallLowering.h"

namespace mlir {
namespace disc_ral {

namespace {

/// Import the GPU Ops to NVVM Patterns.
#include "GPUToNVVM.cpp.inc"

/// A pass that replaces all occurrences of GPU device operations with their
/// corresponding NVVM equivalent.
///
/// This pass only handles device code and is not meant to be run on GPU host
/// code.
struct DiscLowerGpuOpsToNVVMOpsPass
    : public DiscLowerGpuOpsToNVVMOpsPassBase<DiscLowerGpuOpsToNVVMOpsPass> {
  DiscLowerGpuOpsToNVVMOpsPass() = default;
  DiscLowerGpuOpsToNVVMOpsPass(unsigned indexBitwidth) {
    this->indexBitwidth = indexBitwidth;
  }

  void runOnOperation() override {
    gpu::GPUModuleOp m = getOperation();

    // MemRef conversion for GPU to NVVM lowering.
    {
      RewritePatternSet patterns(m.getContext());
      TypeConverter typeConverter;
      typeConverter.addConversion([](Type t) { return t; });
      // NVVM uses alloca in the default address space to represent private
      // memory allocations, so drop private annotations. NVVM uses address
      // space 3 for shared memory. NVVM uses the default address space to
      // represent global memory.
      gpu::populateMemorySpaceAttributeTypeConversions(
          typeConverter, [](gpu::AddressSpace space) -> unsigned {
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
      gpu::populateMemorySpaceLoweringPatterns(typeConverter, patterns);
      ConversionTarget target(getContext());
      gpu::populateLowerMemorySpaceOpLegality(target);
      if (failed(applyFullConversion(m, target, std::move(patterns))))
        return signalPassFailure();
    }

    /// Customize the bitwidth used for the device side index computations.
    LowerToLLVMOptions options(
        m.getContext(),
        DataLayout(cast<DataLayoutOpInterface>(m.getOperation())));
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);

    LLVMTypeConverter converter(m.getContext(), options);
    // Lowering for MMAMatrixType.
    converter.addConversion([&](gpu::MMAMatrixType type) -> Type {
      // The number of items in structToReturn are dependent on the the dataType
      // and the MMA operand that this operation is associated with.
      llvm::DenseMap<StringRef, int64_t> numElemsPerThreadF16,
          numElemsPerThreadF32;
      numElemsPerThreadF16["AOp"] = 8;
      numElemsPerThreadF16["BOp"] = 8;
      numElemsPerThreadF16["COp"] = 4;
      numElemsPerThreadF32["AOp"] = 8;
      numElemsPerThreadF32["BOp"] = 8;
      numElemsPerThreadF32["COp"] = 8;
      Type structToReturn;
      if (type.getElementType().isF16()) {
        // Number of f16's in 32-bit.
        unsigned vecSize = 2;
        Type vec = VectorType::get(vecSize, FloatType::getF16(&getContext()));
        unsigned size = numElemsPerThreadF16[type.getOperand()];
        SmallVector<Type> elements(size, vec);
        structToReturn =
            LLVM::LLVMStructType::getLiteral(&getContext(), elements);
      } else if (type.getElementType().isF32()) {
        unsigned size = numElemsPerThreadF32[type.getOperand()];
        SmallVector<Type> elements(size, FloatType::getF32(&getContext()));
        structToReturn =
            LLVM::LLVMStructType::getLiteral(&getContext(), elements);
      }
      return structToReturn;
    });

    RewritePatternSet patterns(m.getContext());

    // Apply in-dialect lowering first. In-dialect lowering will replace ops
    // which need to be lowered further, which is not supported by a single
    // conversion pass.
    populateGpuRewritePatterns(patterns);
    (void)applyPatternsAndFoldGreedily(m, std::move(patterns));

    RewritePatternSet llvmPatterns(m.getContext());
    populateMathToLLVMConversionPatterns(converter, llvmPatterns);

    // These math ops will be lowered to nvvm intrinsics
    SmallVector<std::string> disablePatterns = {
        "mlir::VectorConvertToLLVMPattern<mlir::math::AbsOp, "
        "mlir::LLVM::FAbsOp>",
        "mlir::VectorConvertToLLVMPattern<mlir::math::CeilOp, "
        "mlir::LLVM::FCeilOp>",
        "mlir::VectorConvertToLLVMPattern<mlir::math::CosOp, "
        "mlir::LLVM::CosOp>",
        "mlir::VectorConvertToLLVMPattern<mlir::math::ExpOp, "
        "mlir::LLVM::ExpOp>",
        "mlir::VectorConvertToLLVMPattern<mlir::math::Exp2Op, "
        "mlir::LLVM::Exp2Op>",
        "{anonymous}::ExpM1OpLowering",
        "mlir::VectorConvertToLLVMPattern<mlir::math::FloorOp, "
        "mlir::LLVM::FFloorOp>",
        "mlir::VectorConvertToLLVMPattern<mlir::math::Log10Op, "
        "mlir::LLVM::Log10Op>",
        "{anonymous}::Log1pOpLowering",
        "mlir::VectorConvertToLLVMPattern<mlir::math::Log2Op, "
        "mlir::LLVM::Log2Op>",
        "mlir::VectorConvertToLLVMPattern<mlir::math::LogOp, "
        "mlir::LLVM::LogOp>",
        "mlir::VectorConvertToLLVMPattern<mlir::math::PowFOp, "
        "mlir::LLVM::PowOp>",
        "{anonymous}::RsqrtOpLowering",
        "mlir::VectorConvertToLLVMPattern<mlir::math::SinOp, "
        "mlir::LLVM::SinOp>",
        "mlir::VectorConvertToLLVMPattern<mlir::math::SqrtOp, "
        "mlir::LLVM::SqrtOp>"};

    // Add the fix before official patterns.
    llvmPatterns.add<GenericAtomicRMWOpLoweringWithBitcast>(
        converter, /* PatternBenefit */ 3);
    llvmPatterns.add<RemoveUselessUnrealizedConversionCastOp>(converter);
    mlir::arith::populateArithToLLVMConversionPatterns(converter, llvmPatterns);
    populateMathToLLVMConversionPatterns(converter, patterns);
    populateFinalizeMemRefToLLVMConversionPatterns(converter, llvmPatterns);
    populateFuncToLLVMConversionPatterns(converter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(converter, llvmPatterns);
    populateGpuToNVVMConversionPatterns(converter, llvmPatterns);
    populateGpuWMMAToNVVMConversionPatterns(converter, llvmPatterns);
    auto llvmFrozenPatterns =
        FrozenRewritePatternSet(std::move(llvmPatterns), disablePatterns, {});
    LLVMConversionTarget target(getContext());
    configureGpuToNVVMConversionLegality(target);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<arith::ArithDialect, math::MathDialect,
                             cf::ControlFlowDialect>();
    target.addIllegalOp<UnrealizedConversionCastOp>();
    if (failed(applyPartialConversion(m, target, llvmFrozenPatterns)))
      signalPassFailure();
  }
};

}  // anonymous namespace

std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
createDiscLowerGpuOpsToNVVMOpsPass(unsigned indexBitwidth) {
  return std::make_unique<DiscLowerGpuOpsToNVVMOpsPass>(indexBitwidth);
}

}  // namespace disc_ral
}  // namespace mlir

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

//===- LowerGpuOpsToROCDLOps.cpp - MLIR GPU to ROCDL lowering passes ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to generate ROCDLIR operations for higher-level
// GPU operations.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/FormatVariadic.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
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
struct GPUShuffleOpLowering : public ConvertOpToLLVMPattern<gpu::ShuffleOp> {
  using ConvertOpToLLVMPattern<gpu::ShuffleOp>::ConvertOpToLLVMPattern;

  /// Lowers a shuffle to the corresponding ROCDL op.
  ///
  /// Convert the `width` argument into an activeMask (a bitmask which specifies
  /// which threads participate in the shuffle) and a maskAndClamp (specifying
  /// the highest lane which participates in the shuffle).
  ///
  ///     %one = llvm.constant(1 : i32) : i32
  ///     %shl = llvm.shl %one, %width : i32
  ///     %active_mask = llvm.sub %shl, %one : i32
  ///     %mask_and_clamp = llvm.sub %width, %one : i32
  ///     %shfl = rocdl.shfl.sync.bfly %active_mask, %value, %offset,
  ///         %mask_and_clamp : float
  ///
  LogicalResult matchAndRewrite(
      gpu::ShuffleOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Location loc = op->getLoc();
    auto valueTy = adaptor.getValue().getType();
    auto int32Type = IntegerType::get(rewriter.getContext(), 32);
    auto predTy = IntegerType::get(rewriter.getContext(), 1);
    Value yes = rewriter.create<LLVM::ConstantOp>(
        loc, predTy, rewriter.getI32IntegerAttr(1));
    Value one = rewriter.create<LLVM::ConstantOp>(
        loc, int32Type, rewriter.getI32IntegerAttr(1));
    // Bit mask of active lanes: `(1 << activeWidth) - 1`.
    Value activeMask = rewriter.create<LLVM::SubOp>(
        loc, int32Type,
        rewriter.create<LLVM::ShlOp>(loc, int32Type, one, adaptor.getWidth()),
        one);
    // Clamp lane: `activeWidth - 1`
    Value maskAndClamp =
        rewriter.create<LLVM::SubOp>(loc, int32Type, adaptor.getWidth(), one);

    Value shfl = rewriter.create<ROCDL::ShflBflyOp>(
        loc, valueTy, activeMask, adaptor.getValue(), adaptor.getOffset(),
        maskAndClamp);

    rewriter.replaceOp(op, {shfl, yes});
    return success();
  }
};

/* Fix atomic codegen problem on AMDMI210, corresponding to the hipcc
 * compilcation results
 */
class AtomicRMWOpRewrite : public OpRewritePattern<LLVM::AtomicRMWOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::AtomicRMWOp atomicOp,
                                PatternRewriter& rewriter) const override {
    if (atomicOp.getOrdering() == LLVM::AtomicOrdering::acq_rel) {
      rewriter.replaceOpWithNewOp<LLVM::AtomicRMWOp>(
          atomicOp, atomicOp.getRes().getType(), atomicOp.getBinOp(),
          atomicOp.getPtr(), atomicOp.getVal(),
          LLVM::AtomicOrdering::monotonic);
    }
    return success();
  }
};

/// Import the GPU Ops to ROCDL Patterns.
#include "GPUToROCDL.cpp.inc"

void configureGpuToROCDLConversionLegality(ConversionTarget& target) {
  target.addIllegalOp<func::FuncOp>();
  target.addLegalDialect<::mlir::LLVM::LLVMDialect>();
  target.addLegalDialect<ROCDL::ROCDLDialect>();
  target.addIllegalDialect<gpu::GPUDialect>();
  target.addIllegalDialect<cf::ControlFlowDialect>();
  target.addIllegalDialect<arith::ArithDialect, math::MathDialect>();
  target.addIllegalOp<LLVM::CosOp, LLVM::ExpOp, LLVM::FAbsOp, LLVM::FCeilOp,
                      LLVM::FFloorOp, LLVM::LogOp, LLVM::Log10Op, LLVM::Log2Op,
                      LLVM::PowOp, LLVM::SinOp, LLVM::SqrtOp>();

  target.addIllegalOp<UnrealizedConversionCastOp>();
  // TODO: Remove once we support replacing non-root ops.
  target.addLegalOp<gpu::YieldOp, gpu::GPUModuleOp, gpu::ModuleEndOp>();
}

void populateGpuToROCDLConversionPatterns(LLVMTypeConverter& converter,
                                          RewritePatternSet& patterns) {
  populateWithGenerated(patterns);
  patterns
      .add<GPUIndexIntrinsicOpLowering<gpu::ThreadIdOp, ROCDL::ThreadIdXOp,
                                       ROCDL::ThreadIdYOp, ROCDL::ThreadIdZOp>,
           GPUIndexIntrinsicOpLowering<gpu::BlockDimOp, ROCDL::BlockDimXOp,
                                       ROCDL::BlockDimYOp, ROCDL::BlockDimZOp>,
           GPUIndexIntrinsicOpLowering<gpu::BlockIdOp, ROCDL::BlockIdXOp,
                                       ROCDL::BlockIdYOp, ROCDL::BlockIdZOp>,
           GPUIndexIntrinsicOpLowering<gpu::GridDimOp, ROCDL::GridDimXOp,
                                       ROCDL::GridDimYOp, ROCDL::GridDimZOp>,
           GPUShuffleOpLowering, GPUReturnOpLowering>(converter);
  patterns.add<GPUFuncOpLowering>(
      converter,
      /*allocaAddrSpace=*/ROCDL::ROCDLDialect::kPrivateMemoryAddressSpace,
      /*workgroupAddrSpace=*/ROCDL::ROCDLDialect::kSharedMemoryAddressSpace,
      StringAttr::get(&converter.getContext(),
                      ROCDL::ROCDLDialect::getKernelFuncAttrName()));
  patterns.add<OpToFuncCallLowering<math::AbsFOp>>(converter, "__ocml_fabs_f32",
                                                   "__ocml_fabs_f64");
  patterns.add<OpToFuncCallLowering<math::AtanOp>>(converter, "__ocml_atan_f32",
                                                   "__ocml_atan_f64");
  patterns.add<OpToFuncCallLowering<math::Atan2Op>>(
      converter, "__ocml_atan2_f32", "__ocml_atan2_f64");
  patterns.add<OpToFuncCallLowering<math::CeilOp>>(converter, "__ocml_ceil_f32",
                                                   "__ocml_ceil_f64");
  patterns.add<OpToFuncCallLowering<math::CosOp>>(converter, "__ocml_cos_f32",
                                                  "__ocml_cos_f64");
  patterns.add<OpToFuncCallLowering<math::CopySignOp>>(
      converter, "__ocml_copysign_f32", "__ocml_copysign_f64");
  patterns.add<OpToFuncCallLowering<math::ExpOp>>(converter, "__ocml_exp_f32",
                                                  "__ocml_exp_f64");
  patterns.add<OpToFuncCallLowering<math::ExpM1Op>>(
      converter, "__ocml_expm1_f32", "__ocml_expm1_f64");
  patterns.add<OpToFuncCallLowering<math::FloorOp>>(
      converter, "__ocml_floor_f32", "__ocml_floor_f64");
  patterns.add<OpToFuncCallLowering<math::LogOp>>(converter, "__ocml_log_f32",
                                                  "__ocml_log_f64");
  patterns.add<OpToFuncCallLowering<math::Log10Op>>(
      converter, "__ocml_log10_f32", "__ocml_log10_f64");
  patterns.add<OpToFuncCallLowering<math::Log1pOp>>(
      converter, "__ocml_log1p_f32", "__ocml_log1p_f64");
  patterns.add<OpToFuncCallLowering<math::Log2Op>>(converter, "__ocml_log2_f32",
                                                   "__ocml_log2_f64");
  patterns.add<OpToFuncCallLowering<math::PowFOp>>(converter, "__ocml_pow_f32",
                                                   "__ocml_pow_f64");
  patterns.add<OpToFuncCallLowering<math::RsqrtOp>>(
      converter, "__ocml_rsqrt_f32", "__ocml_rsqrt_f64");
  patterns.add<OpToFuncCallLowering<math::SinOp>>(converter, "__ocml_sin_f32",
                                                  "__ocml_sin_f64");
  patterns.add<OpToFuncCallLowering<math::SqrtOp>>(converter, "__ocml_sqrt_f32",
                                                   "__ocml_sqrt_f64");
  patterns.add<OpToFuncCallLowering<math::TanhOp>>(converter, "__ocml_tanh_f32",
                                                   "__ocml_tanh_f64");
}

// A pass that replaces all occurrences of GPU device operations with their
// corresponding ROCDL equivalent.
//
// This pass only handles device code and is not meant to be run on GPU host
// code.
struct DiscLowerGpuOpsToROCDLOpsPass
    : public DiscLowerGpuOpsToROCDLOpsPassBase<DiscLowerGpuOpsToROCDLOpsPass> {
  DiscLowerGpuOpsToROCDLOpsPass() = default;
  DiscLowerGpuOpsToROCDLOpsPass(unsigned indexBitwidth) {
    this->indexBitwidth = indexBitwidth;
  }

  void runOnOperation() override {
    gpu::GPUModuleOp m = getOperation();

    m.walk([&](mlir::gpu::GPUFuncOp gpu_kernel) {
      if (gpu_kernel.isKernel()) {
        gpu_kernel->setAttr(
            "rocdl.max_flat_work_group_size",
            mlir::IntegerAttr::get(mlir::IntegerType::get(&getContext(), 32),
                                   1024));
      }
    });

    /// Customize the bitwidth used for the device side index computations.
    LowerToLLVMOptions options(
        m.getContext(),
        DataLayout(cast<DataLayoutOpInterface>(m.getOperation())));
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);
    LLVMTypeConverter converter(m.getContext(), options);

    RewritePatternSet patterns(m.getContext());
    RewritePatternSet llvmPatterns(m.getContext());
    RewritePatternSet postpatterns(m.getContext());

    populateGpuRewritePatterns(patterns);
    (void)applyPatternsAndFoldGreedily(m, std::move(patterns));

    // Add the fix before official patterns.
    llvmPatterns.add<GenericAtomicRMWOpLoweringWithBitcast>(
        converter, /* PatternBenefit */ 3);
    llvmPatterns.add<RemoveUselessUnrealizedConversionCastOp>(converter);
    mlir::arith::populateArithToLLVMConversionPatterns(converter, llvmPatterns);
    populateVectorToLLVMConversionPatterns(converter, llvmPatterns);
    // Remove the following since it disappears from LLVM.
    // populateVectorToROCDLConversionPatterns(converter, llvmPatterns);
    cf::populateControlFlowToLLVMConversionPatterns(converter, llvmPatterns);
    populateMathToLLVMConversionPatterns(converter, patterns);

    populateFinalizeMemRefToLLVMConversionPatterns(converter, llvmPatterns);

    populateFuncToLLVMConversionPatterns(converter, patterns);
    ::mlir::disc_ral::populateGpuToROCDLConversionPatterns(converter,
                                                           llvmPatterns);
    ::mlir::LLVMConversionTarget target(getContext());
    ::mlir::disc_ral::configureGpuToROCDLConversionLegality(target);
    if (failed(applyPartialConversion(m, target, std::move(llvmPatterns))))
      signalPassFailure();

    postpatterns.add<AtomicRMWOpRewrite>(postpatterns.getContext());
    (void)applyPatternsAndFoldGreedily(m, std::move(postpatterns));
  }
};

}  // anonymous namespace

std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
createDiscLowerGpuOpsToROCDLOpsPass(unsigned indexBitwidth) {
  return std::make_unique<DiscLowerGpuOpsToROCDLOpsPass>(indexBitwidth);
}

}  // namespace disc_ral
}  // namespace mlir

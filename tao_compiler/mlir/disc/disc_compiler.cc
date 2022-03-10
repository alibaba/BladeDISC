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

#include "tensorflow/compiler/mlir/disc/disc_compiler.h"

#include <fstream>

#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "mlir-hlo/Dialect/lhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"  // from @llvm-project
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Parser.h"          // from @llvm-project
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "mlir/Support/Timing.h"         // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Export.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/disc/disc_util.h"
#include "tensorflow/compiler/mlir/disc/transforms/codegen_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/fusion_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/passes.h"
#include "tensorflow/compiler/mlir/disc/transforms/placement_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/rewriters.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/compile_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/util/env_var.h"

using mlir::FuncOp;

namespace mlir {
namespace disc_ral {

namespace {

void DumpLLVMModule(llvm::Module* m) {
  std::string str;
  llvm::raw_string_ostream OS(str);
  OS << *m;
  OS.flush();
  llvm::dbgs() << str << "\n";
}

LogicalResult RewriteLLVMModule(llvm::Module* m) {
  auto& ctx = m->getContext();
  auto result_type = llvm::Type::getVoidTy(ctx);
  std::vector<llvm::Type*> arg_types;
  arg_types.push_back(llvm::Type::getInt8Ty(ctx)->getPointerTo());
  arg_types.push_back(llvm::Type::getInt8Ty(ctx)->getPointerTo());
  arg_types.push_back(
      llvm::Type::getInt8Ty(ctx)->getPointerTo()->getPointerTo());
  auto func_type = llvm::FunctionType::get(result_type, arg_types, false);
  auto ral_call = m->getOrInsertFunction("disc_ral_call", func_type);
  llvm::Function* func = llvm::cast<llvm::Function>(ral_call.getCallee());
  auto args = func->arg_begin();
  auto ctx_struct = args++;
  auto api_name = args++;
  auto api_args = args;

  auto block = llvm::BasicBlock::Create(ctx, "entry", func);
  llvm::IRBuilder<> b(block);
  auto typed_ctx_struct = b.CreateBitOrPointerCast(
      ctx_struct, llvm::Type::getInt8Ty(ctx)->getPointerTo()->getPointerTo());
  auto real_ctx = b.CreateLoad(llvm::Type::getInt8Ty(ctx)->getPointerTo(),
                               typed_ctx_struct);
  auto untyped_real_func = b.CreateConstGEP1_32(
      typed_ctx_struct->getType()->getScalarType()->getPointerElementType(),
      typed_ctx_struct, 1);
  auto real_func = b.CreateLoad(
      func_type->getPointerTo(),
      b.CreateBitOrPointerCast(untyped_real_func,
                               func_type->getPointerTo()->getPointerTo()));
  auto first_arg =
      b.CreateLoad(api_args->getType()->getPointerElementType(), api_args);
  auto untyped_first_arg = b.CreateBitOrPointerCast(
      first_arg, llvm::Type::getInt8Ty(ctx)->getPointerTo()->getPointerTo());
  b.CreateStore(real_ctx, untyped_first_arg);
  b.CreateCall(func_type, real_func, {real_ctx, api_name, api_args});
  b.CreateRetVoid();

  return success();
}

std::unique_ptr<llvm::TargetMachine> GetTargetMachine(llvm::Module* module) {
  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    llvm::errs() << "Failed to create a JITTargetMachineBuilder for the host\n";
    return nullptr;
  }
  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) {
    llvm::errs() << "Failed to create a TargetMachine for the host\n";
    return nullptr;
  }
  return std::move(*tmOrError);
}

}  // namespace

CpuLoweringOptions::CpuLoweringOptions(bool init_from_env_vars) {
  if (init_from_env_vars) {
    initFromEnvVars();
  }
}

void CpuLoweringOptions::initFromEnvVars() {
  tensorflow::ReadInt64FromEnvVar("DISC_CPU_FAST_MATH_LEVEL", fast_math_level,
                                  &fast_math_level);
  tensorflow::ReadInt64FromEnvVar("DISC_CPU_VECTOR_WIDTH", vector_width,
                                  &vector_width);
  tensorflow::ReadBoolFromEnvVar("DISC_CPU_DISABLE_LOOP_UNROLL",
                                 disable_loop_unroll, &disable_loop_unroll);
  tensorflow::ReadBoolFromEnvVar("DISC_CPU_ASSUME_NO_BUFFER_ALIAS",
                                 assume_no_buffer_alias,
                                 &assume_no_buffer_alias);
  tensorflow::ReadBoolFromEnvVar("DISC_CPU_ENABLE_MULTI_THREAD",
                                 target_multi_threading,
                                 &target_multi_threading);
}

LogicalResult LowerHLOToLLVM(ModuleOp m, const DISCLoweringOptions& options) {
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  TimingScope timing = tm.getRootScope();
  PassManager pm(m.getContext());
  applyPassManagerCLOptions(pm);
  pm.enableTiming(timing);
  pm.getContext()->disableMultithreading();

  // make sure some dependent dialects are loaded.
  mlir::DialectRegistry registry;
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  mlir::registerROCDLDialectTranslation(registry);
  m.getContext()->appendDialectRegistry(registry);

  bool gpu_enabled = (options.mode == CodeGenMode::kGpuCentric);

  auto printingFlags = OpPrintingFlags();
  printingFlags.elideLargeElementsAttrs(16);
  pm.enableIRPrinting(
      /*shouldPrintBeforePass=*/nullptr,
      /*shouldPrintAfterPass=*/
      [](Pass* pass, Operation*) { return VLOG_IS_ON(1); },
      /*printModuleScope=*/false,
      /*printAfterOnlyOnChange=*/true,
      /*printAfterOnlyOnFailure*/ false, llvm::dbgs(), printingFlags);

  pm.addPass(mlir::createInlinerPass());

  // TODO(disc): Lower HLO shape constraints instead of eliding them here.
  pm.addNestedPass<FuncOp>(disc_ral::createDiscRemoveShapeConstraintsPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());

  // TODO: pay attention if this pass brings side-effects on performance
  // TODO(disc): this pass introduces `tensor.cast(int32_tensor) ->
  // index_tensor`, which is illegal ir. Temporarily disable this pass.
  // TODO(disc): implement another implicit broadcast simplifier pass.
  // pm.addNestedPass<FuncOp>(mhlo::createBroadcastPropagationPass());
  pm.addNestedPass<FuncOp>(createCSEPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());

  // propagate some known shape information.
  pm.addPass(disc_ral::createDiscShapeSimplifierPass());

  pm.addNestedPass<FuncOp>(disc_ral::createDiscConvertTensorToStandardPass());
  pm.addNestedPass<FuncOp>(disc_ral::createDiscConvertHloToStandardPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());

  pm.addNestedPass<FuncOp>(disc_ral::createDiscSplitLargeOpsPass());
  pm.addNestedPass<FuncOp>(disc_ral::createDiscDotRewriterPass());

  // Either merge dots to batched dot or merge dots sharing the same operand.
  pm.addNestedPass<FuncOp>(disc_ral::createDiscDotMergePass());

  if (gpu_enabled) {
    pm.addNestedPass<FuncOp>(mhlo::createHloCanonicalizeReductionPass());
  }

  pm.addPass(disc_ral::createDiscMarkShapeCalcOpPass());
  pm.addPass(disc_ral::createPlacerPass(gpu_enabled));

  // Run CSE after placer pass to eliminate some redundant copies.
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());

  pm.addNestedPass<FuncOp>(disc_ral::createDiscConvRewriter());
  // Run CSE after conv rewriter pass to eliminate some redundant transpose ops.
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(disc_ral::createTransposeSimplifierPass());
  if (gpu_enabled) {
    // Cudnn only supports using same padding value for both left side & right
    // side. This pass ensures this property.
    pm.addNestedPass<FuncOp>(disc_ral::createDiscGpuConvPaddingLegalization());
  }

  // We currently do not support AMP in AICompiler side. If
  // `TAO_MLIR_ENABLE_AMP` is set, we simply convert all gemm ops to fp16.
  //
  // This is a workaround for PyTorch. Some previous Torch version does
  // not support AMP on torch script level. Therefore the exported IR is in
  // the data type of FP32 instead, which makes DISC less likely to benefit
  // in performance if the user's baseline is in FP16 mode.
  bool enable_fp16 = false;
  tensorflow::ReadBoolFromEnvVar("TAO_MLIR_ENABLE_AMP", false, &enable_fp16);
  pm.addNestedPass<FuncOp>(
      disc_ral::createDiscElementTypeConverterPass(enable_fp16));

  // Create tie_shape ops to explicitly express dim size equality info.
  pm.addPass(disc_ral::createDiscShapeSimplifierPass("main", true));
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());

  // TODO(disc): remove this after fixing the bug in hlo-legalize-to-lhlo pass.
  pm.addPass(createFuncBufferizePass());
  pm.addPass(mhlo_disc::createDiscLegalizeToLhloPass());
  pm.addPass(mhlo::createLegalizeToLhloPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());

  // Convert shape to std. Community ```convert-shape-to-std``` pass
  // lowers `shape.broadcast` using scf ops. However, our pass pipeline
  // currently does not support scf well. Thus, we re-write our internal
  // pass to lower shape ops to std dialect.
  pm.addNestedPass<FuncOp>(disc_ral::createDiscConvertShapeToStandardPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(lmhlo::createLegalizeToTensorOpPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());

  // bufferize constant ops & index_cast that have tensor types.
  pm.addNestedPass<FuncOp>(disc_ral::createDiscStdBufferizePass());
  pm.addPass(arith::createArithmeticBufferizePass());
  pm.addNestedPass<FuncOp>(createTensorBufferizePass());
  pm.addNestedPass<FuncOp>(bufferization::createFinalizingBufferizePass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(disc_ral::createDiscMemrefCanonicalizerPass());

  pm.addPass(disc_ral::createDiscAssignMemorySpacePass("main", gpu_enabled));
  pm.addNestedPass<FuncOp>(bufferization::createPromoteBuffersToStackPass(
      [](Value alloc) { return IsSmallCpuAlloc(alloc); }));

  // Enable stitch by default on CPU.
  bool enable_stitch = !gpu_enabled;
  tensorflow::ReadBoolFromEnvVar("DISC_ENABLE_STITCH", !gpu_enabled,
                                 &enable_stitch);
  if (!gpu_enabled) {
    FusionOptions fusionOptions;
    fusionOptions.max_num_arguments_per_kernel = 4096;
    setGlobalFusionOptions(fusionOptions);
  }
  if (enable_stitch && gpu_enabled) {
    // Some passes introduce a bunch of memref alloc, load and store to for
    // shape operations (e.g., ReshapeOp(.., target_shape, ..)), which makes
    // shape equality analysis quite tricky. This pass helps to eliminate
    // unnecessary shape value transformation.
    pm.addNestedPass<FuncOp>(
        disc_ral::createDiscMemRefLoadStoreSimplifierPass());
  }
  pm.addNestedPass<FuncOp>(disc_ral::createDiscFusionPass(
      gpu_enabled, enable_stitch ? "stitch" : "base"));
  if (gpu_enabled) {
    // TODO: Support cpu stitch with splat const
    pm.addNestedPass<FuncOp>(disc_ral::createDiscFuseSplatConstPass());
    auto& gpu_options = options.gpu_info;
    pm.addNestedPass<FuncOp>(
        disc_ral::createDiscSpecializeFusionWithSpeculationPass(
            gpu_options.cc_major, gpu_options.cc_minor));
  } else {
    pm.addNestedPass<FuncOp>(
        disc_ral::createDiscSpecializeFusionWithSpeculationPass());
  }

  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());

  if (enable_stitch && !gpu_enabled) {
    pm.addNestedPass<FuncOp>(disc_ral::createDiscStitchFusionPass());
  }

  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());

  pm.addNestedPass<FuncOp>(bufferization::createBufferDeallocationPass());

  pm.addPass(disc_ral::createRalInjectExecutionContextPass());
  pm.addNestedPass<FuncOp>(disc_ral::createDiscLowerToLibraryCallPass());
  pm.addPass(disc_ral::createDiscConstToRALPass(options.metadata_file_path));

  if (enable_stitch && gpu_enabled) {
    // The passes between `DiscLhloLegalizeRootsToParallelLoops` and
    // 'DiscFusionPass' introduce new shape value transformations with extra
    // memref load and store. Eliminate them.
    pm.addNestedPass<FuncOp>(
        disc_ral::createDiscMemRefLoadStoreSimplifierPass());
  }
  // CodeGen passes: lhlo -> gpu.launch_func
  // TODO: move to aicompiler repo and add more schedules/op coverage
  pm.addNestedPass<FuncOp>(
      disc_ral::createDiscLhloLegalizeRootsToParallelLoopsPass());
  // Converts `atomic_rmw` to `generic_atomic_rmw` when necessary to use CAS.
  pm.addNestedPass<FuncOp>(memref::createExpandOpsPass());
  // Converts `atomic_rmw` to `generic_atomic_rmw` that is unhandled in
  // `StdExpandOps` pass.
  pm.addNestedPass<FuncOp>(
      disc_ral::createDiscUnhandledAtomicRMWConverterPass());
  pm.addNestedPass<FuncOp>(disc_ral::createDiscInputInlineFusionPass());
  pm.addNestedPass<FuncOp>(arith::createArithmeticExpandOpsPass());
  pm.addNestedPass<FuncOp>(memref::createFoldSubViewOpsPass());

  // Flatten multi dim memref accesses to its 1D format to enable more
  // opportunities for linearizeOp/delinearizeOp elimination.
  pm.addNestedPass<FuncOp>(disc_ral::createDiscFlattenMemrefAccessPass());
  // Not using community canonicalizer since it will erase single iteration
  // parallel for loop, which is illegal on GPU backend. Parallel for loops are
  // supposed to be executed on device side while community canonicalizer break
  // this assumption.
  SmallVector<std::string> disablePatterns = {
      "{anonymous}::CollapseSingleIterationLoops",
      "{anonymous}::RemoveEmptyParallelLoops",
      "{anonymous}::MergeNestedParallelLoops"};
  if (!gpu_enabled) {
    disablePatterns.clear();
  }
  pm.addNestedPass<FuncOp>(
      disc_ral::createDiscCanonicalizerPass(disablePatterns));
  pm.addNestedPass<FuncOp>(createCSEPass());
  pm.addNestedPass<FuncOp>(
      disc_ral::createDiscCanonicalizerPass(disablePatterns));
  pm.addNestedPass<FuncOp>(disc_ral::createDiscMemRefCSEPass());
  // convert linearizeOp/delinearizeOp to std dialect.
  pm.addNestedPass<FuncOp>(disc_ral::createDiscConvertShapeToStandardPass());
  pm.addNestedPass<FuncOp>(
      disc_ral::createDiscCanonicalizerPass(disablePatterns));
  pm.addNestedPass<FuncOp>(createCSEPass());
  pm.addNestedPass<FuncOp>(
      disc_ral::createDiscCanonicalizerPass({disablePatterns}));

  if (gpu_enabled) {
    // Coalesce generated parallels to have 1d parallels.
    // TODO: 2D parallel -> collapsing -> tiling process introduces div/rem/if
    // ops which hurts performance. To optimize.
    pm.addNestedPass<FuncOp>(disc_ral::createDiscParallelLoopCollapsingPass());
    // TODO: adopt tileSize from attributes of speculation pass with a
    // wrapper of the original ParallelLoopTilingPass
    pm.addNestedPass<FuncOp>(
        disc_ral::createParallelLoopTilingPass({256}, true));
    pm.addNestedPass<FuncOp>(disc_ral::createMapParallelLoopsPass());

    pm.addNestedPass<FuncOp>(createParallelLoopToGpuPass());
    pm.addPass(createGpuKernelOutliningPass());
    pm.addPass(disc_ral::createDiscAssignKernelNamePass());
  } else {
    if (options.cpu_options.target_multi_threading) {
      pm.addNestedPass<FuncOp>(disc_ral::createDiscCpuMapParallelLoopPass());
      pm.addPass(disc_ral::createDiscOutlineCpuKernelPass());
    }
  }

  pm.addNestedPass<FuncOp>(disc_ral::createLhloFusionInlinerPass());

  if (gpu_enabled) {
    pm.addPass(disc_ral::createReviseGpuKernelOutliningPass());

    // Device side codegen: gpu.module -> cubin
    auto& kernelPm = pm.nest<mlir::gpu::GPUModuleOp>();
    kernelPm.addPass(createConvertSCFToCFPass());
    kernelPm.addPass(createLowerAffinePass());
    kernelPm.addNestedPass<FuncOp>(createCanonicalizerPass());
    kernelPm.addNestedPass<FuncOp>(createCSEPass());
    kernelPm.addNestedPass<FuncOp>(createCanonicalizerPass());
    kernelPm.addPass(createStripDebugInfoPass());
#if TENSORFLOW_USE_ROCM
    kernelPm.addPass(disc_ral::createDiscLowerGpuOpsToROCDLOpsPass(
        /*kDeriveIndexBitwidthFromDataLayout*/ 32));
#elif GOOGLE_CUDA
    kernelPm.addPass(disc_ral::createDiscLowerGpuOpsToNVVMOpsPass(
        /*kDeriveIndexBitwidthFromDataLayout*/ 32));
#endif
    auto& gpu_options = options.gpu_info;
    kernelPm.addPass(disc_ral::CreateDiscGpuKernelToBlobPass(
        gpu_options.cc_major, gpu_options.cc_minor,
        options.gpu_options.multi_cc_support,
        options.gpu_options.multi_cc_support_dbg_ptx_only));
  } else {
    if (options.cpu_options.fast_math_level > 0) {
      // Approximate Tanh using standard operations.
      pm.addNestedPass<FuncOp>(
          mhlo::createLegalizeTrigonometricToApproximationPass());
      pm.addNestedPass<FuncOp>(disc_ral::createDiscMathApproximationPass());
    }
  }

  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());

  pm.addNestedPass<FuncOp>(disc_ral::createDiscRemoveDeadBufferPass());

  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createLowerAffinePass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addPass(createStripDebugInfoPass());

  // Host side codegen: std -> binary
  pm.addPass(disc_ral::createDiscToLLVMPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());

  if (failed(pm.run(m))) return failure();

  TimingScope outputTiming = timing.nest("Output");
  return success();
}

LogicalResult ApplyCpuOptionsBeforeTranslatingToLLVM(
    ModuleOp module, const DISCLoweringOptions& options) {
  auto& cpuOptions = options.cpu_options;
  if (cpuOptions.vector_width > 0) {
    module.walk([&](LLVM::LLVMFuncOp op) {
      MLIRContext* context = op.getContext();
      auto optionName = StringAttr::get(context, "prefer-vector-width");
      auto vectorWidthStr = llvm::Twine(cpuOptions.vector_width).str();
      auto optionValue = StringAttr::get(context, vectorWidthStr);
      auto optionEntry = ArrayAttr::get(context, {optionName, optionValue});
      op->setAttr("passthrough", ArrayAttr::get(context, {optionEntry}));
    });
    if (VLOG_IS_ON(1)) {
      llvm::dbgs() << "[[DISC DEBUG]] avx512 dump begin: \n"
                   << module << "\ndump end\n";
    }
  }

  if (cpuOptions.assume_no_buffer_alias) {
    module.walk([&](LLVM::LLVMFuncOp op) {
      if (!op->getAttrOfType<UnitAttr>(kCpuKernelFunc)) return;
      OpBuilder b(op);
      // The first arg is the ral context, thus we need to skip it.
      for (int i = 1; i < op.getNumArguments(); ++i) {
        if (!op.getArgument(i).getType().isa<LLVM::LLVMPointerType>()) continue;
        op.setArgAttr(i, LLVM::LLVMDialect::getNoAliasAttrName(),
                      b.getUnitAttr());
      }
    });
    if (VLOG_IS_ON(1)) {
      llvm::dbgs() << "[[DISC DEBUG]] noalias dump begin: \n"
                   << module << "\ndump end\n";
    }
  }

  if (cpuOptions.disable_loop_unroll) {
    module.walk([&](Operation* op) {
      if (!isa<LLVM::BrOp, LLVM::CondBrOp>(op)) return;
      OpBuilder opBuilder(op);
      auto keyName = LLVM::LLVMDialect::getLoopOptionsAttrName();
      LLVM::LoopOptionsAttrBuilder b;
      b.setDisableUnroll(true);
      auto valueAttr = LLVM::LoopOptionsAttr::get(op->getContext(), b);
      SmallVector<NamedAttribute> namedAttrs;
      namedAttrs.emplace_back(opBuilder.getNamedAttr(keyName, valueAttr));
      auto dictAttr = DictionaryAttr::get(op->getContext(), namedAttrs);
      op->setAttr(LLVM::LLVMDialect::getLoopAttrName(), dictAttr);
    });
    if (VLOG_IS_ON(1)) {
      llvm::dbgs() << "[[DISC DEBUG]] no_unroll dump begin: \n"
                   << module << "\ndump end\n";
    }
  }

  if (cpuOptions.fast_math_level > 1) {
    module.walk([&](Operation* op) {
      auto fastmathFlagsInterface = dyn_cast<LLVM::FastmathFlagsInterface>(op);
      if (!fastmathFlagsInterface) return;
      LLVM::FastmathFlags fmf;
      switch (cpuOptions.fast_math_level) {
        case 4:
          fmf = fmf | LLVM::FastmathFlags::ninf;
          fmf = fmf | LLVM::FastmathFlags::arcp;
          fmf = fmf | LLVM::FastmathFlags::contract;
          fmf = fmf | LLVM::FastmathFlags::afn;
          fmf = fmf | LLVM::FastmathFlags::fast;
        case 3:
          fmf = fmf | LLVM::FastmathFlags::nnan;
          fmf = fmf | LLVM::FastmathFlags::nsz;
        case 2:
          fmf = fmf | LLVM::FastmathFlags::reassoc;
          break;
        default:
          llvm::errs() << "[[DISC WARNING]] unknown fast_math_level value\n";
          break;
      }
      op->setAttr("fastmathFlags", LLVM::FMFAttr::get(op->getContext(), fmf));
    });
    if (VLOG_IS_ON(1)) {
      llvm::dbgs() << "[[DISC DEBUG]] fastmath dump begin: \n"
                   << module << "\ndump end\n";
    }
  }

  return success();
}

LogicalResult ApplyCpuOptionsAfterTranslatingToLLVM(
    llvm::Module* module, const DISCLoweringOptions& options) {
  // This is only a workaround for select inst. In MLIR LLVM, select inst is not
  // a FastmathFlagsInterface op. We need this to do auto vectorization for
  // reduce max op.
  const auto& fast_math_level = options.cpu_options.fast_math_level;
  if (fast_math_level > 1) {
    llvm::FastMathFlags ffm;
    switch (fast_math_level) {
      case 4:
        ffm.setNoInfs();
        ffm.setAllowReciprocal();
        ffm.setAllowContract();
        ffm.setApproxFunc();
      case 3:
        ffm.setNoNaNs();
        ffm.setNoSignedZeros();
      case 2:
        ffm.setAllowReassoc();
        break;
      default:
        llvm::errs() << "[[DISC WARNING]] unknown fast_math_level value\n";
        break;
    }
    for (auto&& func : module->getFunctionList()) {
      for (auto&& bb : func.getBasicBlockList())
        for (auto&& I : bb) {
          if (!llvm::isa<llvm::SelectInst>(&I)) continue;
          I.setFastMathFlags(ffm);
        }
    }
  }
  return success();
}

LogicalResult LowerLLVMToBinary(ModuleOp module,
                                const DISCLoweringOptions& options,
                                std::string& out) {
  bool gpu_enabled = (options.mode == CodeGenMode::kGpuCentric);
  if (!gpu_enabled &&
      failed(ApplyCpuOptionsBeforeTranslatingToLLVM(module, options))) {
    llvm::errs() << "failed to apply cpu options before translating to llvm\n";
    return failure();
  }

  // Translate the module.
  llvm::LLVMContext llvm_context;
  mlir::registerLLVMDialectTranslation(*module->getContext());
  std::unique_ptr<llvm::Module> llvm_module =
      mlir::translateModuleToLLVMIR(module, llvm_context);

  if (VLOG_IS_ON(1)) {
    llvm::dbgs() << "before optimize llvm module:\n";
    DumpLLVMModule(llvm_module.get());
  }

  std::unique_ptr<llvm::TargetMachine> tm = GetTargetMachine(llvm_module.get());
  if (!tm) {
    llvm::errs() << "create TargetMachine failed\n";
    return failure();
  }

  if (!gpu_enabled && failed(ApplyCpuOptionsAfterTranslatingToLLVM(
                          llvm_module.get(), options))) {
    llvm::errs() << "failed to apply cpu options after translating to llvm\n";
    return failure();
  }

  if (failed(RewriteLLVMModule(llvm_module.get()))) {
    llvm::errs() << "rewrite llvm module failed\n";
    return failure();
  }

  auto transformer = mlir::makeOptimizingTransformer(
      /*optLevel=*/3, /*sizeLevel=*/0, /*targetMachine=*/tm.get());
  if (!transformer) {
    llvm::errs() << "transformer create failed\n";
    return failure();
  }

  if (transformer(llvm_module.get())) {
    llvm::errs() << "apply transformer failed\n";
    return failure();
  }

  if (VLOG_IS_ON(1)) {
    llvm::dbgs() << "after optimize llvm module:\n";
    DumpLLVMModule(llvm_module.get());
  }

  // Set up the output stream.
  llvm::SmallString<8> outstr;
  llvm::raw_svector_ostream ostream(outstr);
  ostream.SetUnbuffered();

  llvm::legacy::PassManager codegen_passes;
  codegen_passes.add(new llvm::TargetLibraryInfoWrapperPass(
      llvm::Triple(llvm_module->getTargetTriple())));

  if (tm->addPassesToEmitFile(codegen_passes, ostream, nullptr,
                              llvm::CGFT_ObjectFile, false)) {
    llvm::errs() << "Failed add passes to emit file\n";
    return failure();
  }
  codegen_passes.run(*llvm_module);
  out = ostream.str().str();

  if (VLOG_IS_ON(2)) {
    llvm::errs() << "binary str: " << out << "\n";
  }

  return success();
}

LogicalResult BinaryStrToSharedLibrary(const DISCLoweringOptions& options,
                                       const std::string& binary) {
  std::string object_filename = options.output_file_name + ".o";

  // TODO: ONLY used for quick testing.
  std::string cmd =
      "gcc --shared -o " + options.output_file_name + " " + object_filename;

  if (VLOG_IS_ON(1)) {
    llvm::errs() << "object file to shared library command: " << cmd << "\n";
  }

  std::ofstream out(object_filename, std::ios::out | std::ios::binary);
  out << binary;
  out.close();

  std::system(cmd.c_str());
  std::remove(object_filename.c_str());

  if (VLOG_IS_ON(1)) {
    llvm::errs() << "save shared lib file to : " << options.output_file_name
                 << "\n";
  }

  return success();
}

LogicalResult LowerHLOToSharedLibrary(ModuleOp m,
                                      const DISCLoweringOptions& options) {
  if (failed(LowerHLOToLLVM(m, options))) {
    llvm::errs() << "lower hlo to llvm failed\n";
    return failure();
  }

  std::string binary;
  if (failed(LowerLLVMToBinary(m, options, binary))) {
    llvm::errs() << "lower llvm to binary failed\n";
    return failure();
  }

  if (failed(BinaryStrToSharedLibrary(options, binary))) {
    llvm::errs() << "lower binary to shared library failed\n";
    return failure();
  }

  return success();
}

}  // namespace disc_ral
}  // namespace mlir

namespace tensorflow {

Status ConvertTF2MlirHlo(mlir::ModuleOp module_op) {
  mlir::PassManager tf2xla(module_op.getContext());

  tf2xla.getContext()->disableMultithreading();
  tf2xla.enableIRPrinting(
      /*shouldPrintBeforePass=*/nullptr,
      /*shouldPrintAfterPass=*/
      [](mlir::Pass* pass, mlir::Operation*) { return VLOG_IS_ON(1); },
      /*printModuleScope=*/false,
      /*printAfterOnlyOnChange=*/true,
      /*printAfterOnlyOnFailure*/ false, llvm::dbgs());

  tf2xla.addPass(mlir::disc_ral::createReviseArgsForStaticRankPass());

  mlir::SmallVector<std::unique_ptr<mlir::Pass>, 1> lower_passes;
  lower_passes.emplace_back(mlir::disc_ral::createDiscLowerTfPass());
  CreateConvertMlirToXlaHloPipeline(
      tf2xla, "XLA_CPU_JIT",
      /*prefer_tf2xla=*/false,
      /*custom_legalization_passes=*/lower_passes);

  // Make sure we catch any error reported by MLIR and forward it to the TF
  // error reporting system. Report a generic error if pass manager failed
  // without emitting a diagnostic.
  mlir::StatusScopedDiagnosticHandler error_handler(module_op.getContext());

  if (failed(tf2xla.run(module_op))) {
    return error_handler.Combine(
        errors::Internal("MLIR TF to XLA legalization failed"));
  }

  return Status::OK();
}

}  // namespace tensorflow

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
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"   // from @llvm-project
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
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/compile_mlir_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/xla/transforms/adjust_layout.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/core/util/env_var.h"

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
      llvm::Type::getInt8Ty(ctx)->getPointerTo(), typed_ctx_struct, 1);
  auto real_func = b.CreateLoad(
      func_type->getPointerTo(),
      b.CreateBitOrPointerCast(untyped_real_func,
                               func_type->getPointerTo()->getPointerTo()));
  auto first_arg =
      b.CreateLoad(llvm::Type::getInt8Ty(ctx)->getPointerTo(), api_args);
  auto untyped_first_arg = b.CreateBitOrPointerCast(
      first_arg, llvm::Type::getInt8Ty(ctx)->getPointerTo()->getPointerTo());
  b.CreateStore(real_ctx, untyped_first_arg);
  b.CreateCall(func_type, real_func, {real_ctx, api_name, api_args});
  b.CreateRetVoid();

  return success();
}

std::unique_ptr<llvm::TargetMachine> GetTargetMachine(llvm::Module* module) {
  llvm::Triple triple(module->getTargetTriple());
  if (triple.getTriple().empty()) {
    triple = llvm::Triple(llvm::sys::getDefaultTargetTriple());
    module->setTargetTriple(triple.getTriple());
  }
  if (VLOG_IS_ON(1)) {
    llvm::errs() << "host default target triple: "
                 << llvm::sys::getDefaultTargetTriple() << "\n";
    llvm::errs() << "host cpu name: " << llvm::sys::getHostCPUName() << "\n";
  }

  std::string error;
  const llvm::Target* target =
      llvm::TargetRegistry::lookupTarget("", triple, error);
  if (!target) {
    return nullptr;
  }

  // Retrieve host CPU name and sub-target features and add them to builder.
  // Relocation model, code model and codegen opt level are kept to default
  // values.
  llvm::StringMap<bool> FeatureMap;
  llvm::SubtargetFeatures Features;
  llvm::sys::getHostCPUFeatures(FeatureMap);
  for (auto& Feature : FeatureMap)
    Features.AddFeature(Feature.first(), Feature.second);

  return std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
      triple.str(), std::string(llvm::sys::getHostCPUName()),
      Features.getString(), llvm::TargetOptions(), llvm::Reloc::Model::PIC_));
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
  using mlir::func::FuncOp;

  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  // Records elapsed time for each pass in the passpipe
  tm.setEnabled(true);
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
  pm.addNestedPass<FuncOp>(createCSEPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());

  bool enable_shape_constraint_ir = useShapeConstraintIR();
  if (!enable_shape_constraint_ir) {
    // propagate some known shape information.
    pm.addPass(disc_ral::createDiscShapeSimplifierPass());
  } else {
    pm.addNestedPass<FuncOp>(disc_ral::createDiscConvertShapeToStandardPass());
    // shape-related optimization
    pm.addPass(disc_ral::createDiscShapeOptimizationPass());
  }

  pm.addNestedPass<FuncOp>(disc_ral::createDiscConvertTensorToStandardPass());
  pm.addNestedPass<FuncOp>(disc_ral::createDiscConvertHloToStandardPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());

  pm.addNestedPass<FuncOp>(disc_ral::createDiscAlgebraSimplifierPass());
  pm.addNestedPass<FuncOp>(disc_ral::createDiscSplitLargeOpsPass());
  pm.addNestedPass<FuncOp>(disc_ral::createDiscDotRewriterPass());
  if (enable_shape_constraint_ir) {
    // shape-related optimization
    pm.addPass(disc_ral::createDiscShapeOptimizationPass());
  }

  // When `DISC_ENABLE_SPARSE` is set, we will find dense weight and convert
  // it to sparse if its meet condition.
  bool enable_sparse = false;
  tensorflow::ReadBoolFromEnvVar("DISC_ENABLE_SPARSE", false, &enable_sparse);
  if (!enable_sparse) {
    // Either merge dots to batched dot or merge dots sharing the same operand.
    pm.addNestedPass<FuncOp>(disc_ral::createDiscDotMergePass());
  }

  if (enable_shape_constraint_ir) {
    // shape-related optimization
    pm.addPass(disc_ral::createDiscShapeOptimizationPass());
  }

  if (gpu_enabled) {
    pm.addNestedPass<FuncOp>(mhlo::createHloCanonicalizeReductionPass());
    if (enable_shape_constraint_ir) {
      // shape-related optimization
      pm.addPass(disc_ral::createDiscShapeOptimizationPass());
    }
  }

  pm.addPass(disc_ral::createDiscMarkShapeCalcOpPass());
  pm.addPass(disc_ral::createPlacerPass(gpu_enabled));

  // Run CSE after placer pass to eliminate some redundant copies.
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());

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
  if (enable_shape_constraint_ir) {
    // shape-related optimization
    pm.addPass(disc_ral::createDiscShapeOptimizationPass());
  }

  if (enable_sparse) {
    pm.addNestedPass<FuncOp>(disc_ral::createDiscDenseToSparsePass());
  }

  auto& gpu_options = options.gpu_info;
  pm.addNestedPass<FuncOp>(disc_ral::createDiscConvRewriter(
      gpu_options.cc_major, gpu_options.cc_minor));
  if (enable_shape_constraint_ir) {
    // shape-related optimization
    pm.addPass(disc_ral::createDiscShapeOptimizationPass());
  }
  // Run CSE after conv rewriter pass to eliminate some redundant transpose ops.
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(disc_ral::createTransposeSimplifierPass());

  if (enable_sparse) {
    pm.addNestedPass<FuncOp>(
        disc_ral::createDiscSparseGemmTransposeSimplifierPass());
  }

  if (gpu_enabled) {
    // Cudnn only supports using same padding value for both left side & right
    // side. This pass ensures this property.
    pm.addNestedPass<FuncOp>(disc_ral::createDiscGpuConvPaddingLegalization());
  }
  if (enable_shape_constraint_ir) {
    // shape-related optimization
    pm.addPass(disc_ral::createDiscShapeOptimizationPass());
    pm.addNestedPass<FuncOp>(disc_ral::createDiscAlgebraSimplifierPass());
  }

  if (!enable_shape_constraint_ir) {
    // Create tie_shape ops to explicitly express dim size equality info.
    pm.addPass(disc_ral::createDiscShapeSimplifierPass("main", true));
  } else {
    // shape-related optimization
    // Create tie_shape ops to explicitly express dim size equality info.
    pm.addPass(disc_ral::createDiscShapeOptimizationPass("main", true));
  }
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());

  // TODO(disc): remove this after fixing the bug in hlo-legalize-to-lhlo pass.
  pm.addPass(func::createFuncBufferizePass());
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

  // Enable stitch by default on CPU.
  bool enable_stitch = !gpu_enabled;
  tensorflow::ReadBoolFromEnvVar("DISC_ENABLE_STITCH", !gpu_enabled,
                                 &enable_stitch);
  if (enable_shape_constraint_ir) {
    pm.addNestedPass<FuncOp>(
        disc_ral::createDiscDuplicateComputationForFusionPass(
            gpu_enabled, enable_stitch ? "stitch" : "base"));
  }
  pm.addNestedPass<FuncOp>(bufferization::createPromoteBuffersToStackPass(
      [](Value alloc) { return IsSmallCpuAlloc(alloc); }));

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
            gpu_options.sm_count, gpu_options.max_threads_per_sm));
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
  if (gpu_enabled && isMemIntensiveOptExperimentalEnabled()) {
    // More benchmarks are on the way to evaluate the effectiveness of this
    // optimization. Then this pass will be enabled by default.
    pm.addNestedPass<FuncOp>(disc_ral::createForLoopUnrollInterleavePass());
  }
  pm.addNestedPass<FuncOp>(arith::createArithmeticExpandOpsPass());
  pm.addNestedPass<FuncOp>(mlir::memref::createFoldMemRefAliasOpsPass());

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
    // pm.addNestedPass<FuncOp>(disc_ral::createMapParallelLoopsPass());
    pm.addNestedPass<FuncOp>(mlir::createGpuMapParallelLoopsPass());

    pm.addNestedPass<FuncOp>(createParallelLoopToGpuPass());
    pm.addPass(createGpuLauchSinkIndexComputationsPass());
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
    if (isMemIntensiveOptExperimentalEnabled()) {
      // We do not enable the LICM pass by default because this function is not
      // widely used and we worry about the robustness. It is not guaranteed to
      // be bug-free. We will enable it by default after evaluation on more
      // benchmarks.
      kernelPm.addPass(createLoopInvariantCodeMotionPass());
      kernelPm.addNestedPass<gpu::GPUFuncOp>(
          createSideEffectLoopInvariantCodeMotionPass());
      // Do LICM again after the above side-affect-LICM to enable more
      // optimizations.
      kernelPm.addPass(createLoopInvariantCodeMotionPass());
      kernelPm.addPass(createCSEPass());
    }
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
    if (isMemIntensiveOptExperimentalEnabled()) {
      // To eliminate dead argument of GPU LLVM functions. First, it has to
      // simplify InsertValueOp and ExtractValueOp which has fake dependency
      // on function parameters. Second, eliminate the dead arguments and append
      // attributes telling the indices of the eliminated arguments, which will
      // be used for generating ral kernel-launch logic of the gpu functions.
      kernelPm.addNestedPass<LLVM::LLVMFuncOp>(
          disc_ral::createLLVMInsertValueSimplifierPass());
      kernelPm.addPass(disc_ral::createFunctionDeadArgumentEliminationPass());
    }
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

  if (enable_shape_constraint_ir) {
    // remove disc_shape.SymbolicDim ops and related ops.
    pm.addPass(disc_ral::createStripShapeConstraintOpsPass());
  }

  // Host side codegen: std -> binary
  pm.addPass(disc_ral::createDiscToLLVMPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());

  if (failed(pm.run(m))) return failure();
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

  if (tm->getRelocationModel() != llvm::Reloc::Model::PIC_) {
    llvm::errs() << "-fPIC must be specified\n";
    return failure();
  }
  llvm_module->setDataLayout(tm->createDataLayout());

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
  mlir::PassManager pm(module_op.getContext());

  pm.getContext()->disableMultithreading();
  auto printingFlags = mlir::OpPrintingFlags();
  printingFlags.elideLargeElementsAttrs(16);
  pm.enableIRPrinting(
      /*shouldPrintBeforePass=*/nullptr,
      /*shouldPrintAfterPass=*/
      [](mlir::Pass* pass, mlir::Operation*) { return VLOG_IS_ON(1); },
      /*printModuleScope=*/false,
      /*printAfterOnlyOnChange=*/true,
      /*printAfterOnlyOnFailure*/ false, llvm::dbgs(), printingFlags);
  bool prefer_tf2xla = false;
  llvm::StringRef device_type = "XLA_CPU_JIT";

  // Replace const arguments to ConstOp and update argument type if it is a
  // fixed-shaped input
  pm.addPass(mlir::disc_ral::createReviseArgsForStaticRankPass());

  // Note that the region-based control-flow produced here still contains
  // function call ops which get inlined by the subsequent inliner pass.
  pm.addPass(mlir::TF::CreateTFFunctionalControlFlowToRegions());
  pm.addPass(mlir::createInlinerPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::TF::CreateDropWhileShapeInvariantPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  // The SCCP pass performs constant propagation across the IR, which, for
  // example, propagates constant arguments into callee functions.
  // TOOD(hinsu): Investigate if we really need SCCP pass before shape inference
  // and can do with just one pass after the shape inference.
  pm.addPass(mlir::createSCCPPass());
  // Guarantee all functions have one use, which enables shape inference.
  pm.addPass(mlir::TF::CreateGuaranteeAllFuncsOneUsePass());
  // Run shape inference pass before tensorlist decomposition to get buffer
  // shape of uninitialized TensorLists.
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());

  // Run SCCP pass again as the availability of shapes may open up new
  // opportunities for constant propagation. Note that the shape inference pass
  // doesn't materialize new constants even if those are computed internally for
  // the purpose of shape inference. These constants might be required by the
  // legalization passes.
  pm.addPass(mlir::createSCCPPass());

  pm.addPass(mlir::TF::CreateTensorListOpsDecompositionPass());
  pm.addPass(mlir::TF::CreateStackOpsDecompositionPass());
  pm.addPass(mlir::TF::CreateTensorArrayOpsDecompositionPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::TFDevice::CreateDecomposeResourceOpsPass());
  pm.addPass(mlir::TF::CreatePromoteResourcesToArgsPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());
  // TODO(b/171426148): We cannot completely remove region to functional control
  // flow conversion from this pipeline yet as it causes some unit tests to
  // fail.
  pm.addPass(mlir::TF::CreateTFRegionControlFlowToFunctional());
  // LegalizeTFControlFlow encapsulates arguments for control flow operations
  // with a tuple argument which break the assumption of resource lifting
  // inside PromoteResourcesToArgs.
  pm.addPass(mlir::mhlo::createLegalizeTFControlFlowPass());

  // customized tf2mhlo converters of DISC
  pm.addNestedPass<mlir::func::FuncOp>(mlir::disc_ral::createDiscLowerTfPass());

  pm.addNestedPass<mlir::func::FuncOp>(mlir::TF::CreateLowerQuantizedPass());
  pm.addPass(mlir::mhlo::CreateLegalizeTfTypesPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::mhlo::createLegalizeTFPass(
      /*allow_partial_conversion=*/true, /*legalize_chlo=*/true,
      /*tf2xla_fallback_device_type=*/device_type, prefer_tf2xla));

  // customized tf2mhlo converters of DISC
  pm.addNestedPass<mlir::func::FuncOp>(mlir::disc_ral::createDiscLowerTfPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::mhlo::CreateAdjustLayoutPass());
  pm.addPass(mlir::mhlo::CreateLegalizeTFCommunicationPass());
  pm.addPass(mlir::mhlo::CreateLegalizeTFCollectivePass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  // Run shape inference pass to propagate shapes through tensor_cast operations
  // from static to dynamic shapes. This could be generated if the shape
  // inference was originally missing in a TF op but the corresponding HLO op
  // had static shape after lowering.
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());
  // Run LegalizeTFPass again because the previous legalization passes can
  // expose more graph pruning and canonicalization opportunities that are
  // necessary for the second LegalizeTFPass(allow_partial_conversion=false)
  // invocation.
  pm.addNestedPass<mlir::func::FuncOp>(mlir::mhlo::createLegalizeTFPass(
      /*allow_partial_conversion=*/false, /*legalize_chlo=*/true,
      /*tf2xla_fallback_device_type=*/device_type, prefer_tf2xla));

  // In order to export to XLA, we must sink constants to control flow regions,
  // since XLA uses functional control flow.
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createSinkConstantsToControlFlowPass());

  // Make sure we catch any error reported by MLIR and forward it to the TF
  // error reporting system. Report a generic error if pass manager failed
  // without emitting a diagnostic.
  mlir::StatusScopedDiagnosticHandler error_handler(module_op.getContext());

  if (failed(pm.run(module_op))) {
    return error_handler.Combine(
        errors::Internal("MLIR TF to XLA legalization failed"));
  }

  return Status::OK();
}

}  // namespace tensorflow

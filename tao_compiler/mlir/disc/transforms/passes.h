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

#ifndef DISC_TRANSFORMS_PASSES_H_
#define DISC_TRANSFORMS_PASSES_H_

#include <memory>

#include "llvm/ADT/ArrayRef.h"

namespace mlir {

class FuncOp;
class FunctionPass;
class ModuleOp;
class Operation;
template <typename T>
class OperationPass;
class Pass;

namespace gpu {
class GPUModuleOp;
}

namespace disc_ral {

// Lower disc to llvm dialect
std::unique_ptr<OperationPass<ModuleOp>> createDiscToLLVMPass();

// Split ops with too many operands, which may exceed the maximum parameter
// space of a GPU kernel before splitting.
std::unique_ptr<OperationPass<FuncOp>> createDiscSplitLargeOpsPass(
    int max_num_operands_per_op = 32);

// Replace const arguments to ConstOp and update argument type if it is a
// fixed-shaped input
std::unique_ptr<OperationPass<ModuleOp>> createReviseArgsForStaticRankPass();

// Lowers the roots of lmhlo.fusion to parallel loops
std::unique_ptr<OperationPass<FuncOp>>
createDiscLhloLegalizeRootsToParallelLoopsPass(int sm_count = -1,
                                               int cc_major = 8,
                                               int cc_minor = 0);

// Canonicalize conv ops to be suitable for lowering to cudnn lib calls.
std::unique_ptr<OperationPass<FuncOp>> createDiscConvRewriter(int cc_major = 8,
                                                              int cc_minor = 0);

// Canonicalize qconv ops to be suitable for lowering to cudnn lib calls.
std::unique_ptr<OperationPass<FuncOp>> createDiscQuantizedConvRewriter(
    int cc_major = 8, int cc_minor = 0);

std::unique_ptr<OperationPass<FuncOp>> createDiscQuantizedDotRewriter();

// Rewrite and decompose mhlo ops
std::unique_ptr<OperationPass<FuncOp>>
createDiscMhloDecompositionRewriterPass();

// Rewrite mhlo::ReduceOp
std::unique_ptr<OperationPass<FuncOp>> createDiscReductionRewriterPass();

// Rewrite dot to fold transpose.
std::unique_ptr<OperationPass<FuncOp>> createDiscDotRewriterPass();

// Convert conv ops' padding value to be suitable for lowering to cudnn lib
// calls.
std::unique_ptr<OperationPass<FuncOp>> createDiscGpuConvPaddingLegalization();

// A pass to specialize fusion op with some speculation.
std::unique_ptr<OperationPass<FuncOp>>
createDiscSpecializeFusionWithSpeculationPass(int sm_count = -1,
                                              int max_threads_per_sm = -1);

// Eliminates certain element types as the input or output of ops by inserting
// Convert ops.
std::unique_ptr<OperationPass<FuncOp>> createDiscElementTypeConverterPass(
    bool enable_fp16_gemm = false, bool enable_fp16_conv = false,
    bool promote_fp16_sensitive_ops_to_f32 = false);

// Greedily maps loops to GPU hardware dimensions.
// TODO: this pass is only a wrapper to mlir func, copied from
// tools/kernel_gen/transforms Should consider of adding this pass to mlir repo.
// std::unique_ptr<OperationPass<FuncOp>> createMapParallelLoopsPass();

// Collapses ParallelsOp into 1-D.
std::unique_ptr<OperationPass<FuncOp>> createDiscParallelLoopCollapsingPass();

// This pass is a revision to the pass in MLIR repo.
// The PR is on reviewing at https://reviews.llvm.org/D105455?id=356595
// TODO: remove this pass when the PR is merged.
std::unique_ptr<OperationPass<FuncOp>> createParallelLoopTilingPass(
    llvm::ArrayRef<int64_t> tileSize = {}, bool withInboundCheck = false);

// Fuse lmhlo ops to kLoop/kInput fusion patterns
std::unique_ptr<OperationPass<FuncOp>> createDiscFusionPass(
    bool gpu_enabled = true, const std::string& fusion_strategy = "base",
    bool mlir_compute_intensive_codegen = false);

// Fuse splat constant to avoid memory access
std::unique_ptr<OperationPass<FuncOp>> createDiscFuseSplatConstPass();

// Mark shape calculating Ops.
std::unique_ptr<OperationPass<ModuleOp>> createDiscMarkShapeCalcOpPass();

// A pass convert lmhlo ConstOp to ral_const_gpu or ral_const_cpu
std::unique_ptr<OperationPass<ModuleOp>> createDiscConstToRALPass(
    const std::string& metadata_file_path = "metadata.pbtxt");

// Place shape calculating Ops.
std::unique_ptr<OperationPass<ModuleOp>> createPlacerPass(bool on_gpu = true);

// Revise the kernel outlining by expanding the host memref into scalars
std::unique_ptr<OperationPass<ModuleOp>> createReviseGpuKernelOutliningPass();

// Lower some specific ops to library calls (modeled by `disc_ral.launch` op).
std::unique_ptr<OperationPass<FuncOp>> createDiscLowerToLibraryCallPass(
    bool gpu_enabled = true);

// Assign memory space tag for each memref type.
std::unique_ptr<OperationPass<ModuleOp>> createDiscAssignMemorySpacePass(
    const std::string& entry_func_name = "main", bool gpu_enabled = true);

std::unique_ptr<OperationPass<gpu::GPUModuleOp>> CreateDiscGpuKernelToBlobPass(
    int cc_major = 7, int cc_minor = 5, bool multi_cc_support = false,
    bool multi_cc_support_dbg_ptx_only = false,
    mlir::StringRef blob_annotation = "gpu.binary");

// convert shape ops to std dialect
std::unique_ptr<OperationPass<FuncOp>> createDiscConvertShapeToStandardPass();

// convert tensor ops to std dialect
std::unique_ptr<OperationPass<FuncOp>> createDiscConvertTensorToStandardPass();

// Fuse the rest of the lmhlo nodes into the parallel loops after the roots
// has been expanded into loops in LhloLegalizeRootsToParallelLoops
std::unique_ptr<OperationPass<FuncOp>> createDiscInputInlineFusionPass();

// Remove all shape constraint ops.
std::unique_ptr<OperationPass<FuncOp>> createDiscRemoveShapeConstraintsPass();

// Convert some hlo ops that are used to do shape computation to std dialect.
std::unique_ptr<OperationPass<FuncOp>> createDiscConvertHloToStandardPass();

// Bufferize std tensor constant.
std::unique_ptr<OperationPass<FuncOp>> createDiscStdBufferizePass();

// CSE of memref.load specific for DISC.
std::unique_ptr<OperationPass<FuncOp>> createDiscMemRefCSEPass();

// Simplifier of memref.load/store specific for DISC.
std::unique_ptr<OperationPass<FuncOp>>
createDiscMemRefLoadStoreSimplifierPass();

// Lowering some tensorflow ops.
std::unique_ptr<OperationPass<FuncOp>> createDiscLowerTfPass(
    const std::string& pdll_files = "",
    const std::string& pdll_include_dirs = "");

// A pass to remove buffers that are not accessed by others
std::unique_ptr<OperationPass<FuncOp>> createDiscRemoveDeadBufferPass();

// Assign a meaningful name to each gpu kernel.
std::unique_ptr<OperationPass<ModuleOp>> createDiscAssignKernelNamePass();

// Lower GPU ops to NVVM ops. Deal with GenericAtomicRMW specifically.
std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
createDiscLowerGpuOpsToNVVMOpsPass(unsigned indexBitwidth = 0);

// Lower GPU ops to ROCDL ops.
std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
createDiscLowerGpuOpsToROCDLOpsPass(unsigned indexBitwidth = 0);

// Convert unhandled complex atomic_rmw ops into generic_atomic_rmw ops.
std::unique_ptr<OperationPass<ModuleOp>>
createDiscUnhandledAtomicRMWConverterPass();

// propagate some known shape information.
std::unique_ptr<OperationPass<ModuleOp>> createDiscShapeSimplifierPass(
    const std::string& entry_func_name = "main", bool insert_tie_shape = false);

// Using approximation impl for some special math ops.
std::unique_ptr<OperationPass<FuncOp>> createDiscMathApproximationPass();

// Flatten memref access to its 1D format.
std::unique_ptr<OperationPass<FuncOp>> createDiscFlattenMemrefAccessPass();

// A canonicalizer with whitelist/blacklist support.
std::unique_ptr<Pass> createDiscCanonicalizerPass(
    const SmallVector<std::string>& disabledPatterns = {},
    const SmallVector<std::string>& enabledPatterns = {});

// Do some memref-related cleanup.
std::unique_ptr<OperationPass<FuncOp>> createDiscMemrefCanonicalizerPass();

// outline each cpu kernel to a dedicated function
std::unique_ptr<OperationPass<ModuleOp>> createDiscOutlineCpuKernelPass();

// assign a parallel schedule for each parallel op on cpu
std::unique_ptr<OperationPass<FuncOp>> createDiscCpuMapParallelLoopPass();

// lowering kStitch fusion pattern to parallel loops.
std::unique_ptr<OperationPass<FuncOp>> createDiscStitchFusionPass();

// Remove some redundant transpose ops.
std::unique_ptr<OperationPass<FuncOp>> createTransposeSimplifierPass();

// Inject disc_ral context into the entry function.
std::unique_ptr<OperationPass<ModuleOp>> createRalInjectExecutionContextPass(
    const std::string& entry_func_name = "main");

// inline lmhlo.Fusion
std::unique_ptr<OperationPass<FuncOp>> createLhloFusionInlinerPass();

// Dot merge optimization.
std::unique_ptr<OperationPass<FuncOp>> createDiscDotMergePass();

// Quantized dot merge optimization.
std::unique_ptr<OperationPass<FuncOp>> createDiscQuantizedDotMergePass();

// Apply some basic algebra simplification optimizations.
std::unique_ptr<OperationPass<FuncOp>> createDiscAlgebraicSimplifierPass();

// apply some shape-related optimization.
std::unique_ptr<OperationPass<ModuleOp>> createDiscShapeOptimizationPass(
    const std::string& entry_func_name = "main", bool keep_tie_shape = false);

// strip disc shape constraint related ops.
std::unique_ptr<OperationPass<ModuleOp>> createStripShapeConstraintOpsPass();

// SCF for-loop unrolling and interleaving.
std::unique_ptr<OperationPass<FuncOp>> createForLoopUnrollInterleavePass();

std::unique_ptr<OperationPass<func::FuncOp>>
createDiscDuplicateComputationForFusionPass(
    bool gpu_enabled = true, const std::string& fusion_strategy = "base");

// Eliminate unused InsertValueOp inside LLVMFuncOp.
std::unique_ptr<OperationPass<LLVM::LLVMFuncOp>>
createLLVMInsertValueSimplifierPass();

// Eliminate dead arguments of GPU LLVM functions.
std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
createFunctionDeadArgumentEliminationPass();

// Apply loop invariant code motion on operators with side effect.
std::unique_ptr<OperationPass<gpu::GPUFuncOp>>
createSideEffectLoopInvariantCodeMotionPass();

// Apply dense gemm/conv to sparse gemm/conv optimizations.
std::unique_ptr<OperationPass<FuncOp>> createDiscDenseToSparsePass(
    bool enable_sparse_convert = false);

// Remove some redundant transpose ops before sparse gemm/conv.
std::unique_ptr<OperationPass<FuncOp>>
createDiscSparseGemmTransposeSimplifierPass();

// Epand ext/trunc of bfloat16
std::unique_ptr<OperationPass<func::FuncOp>> createDiscBF16ExpansionPass();

// Converts fake_quant annotated graph to the real quantized version.
std::unique_ptr<OperationPass<func::FuncOp>> createDiscConvertFakeQuantOpPass();

// Lowers quantize and dequantize ops to a bunch of basic elementwise ops.
std::unique_ptr<OperationPass<func::FuncOp>>
createDiscLowerQuantizeAndDequantizePass();

// transform weight data layout for ft's weight-only qgemm.
std::unique_ptr<OperationPass<func::FuncOp>>
createDiscTranformWeightDataLayoutForWeightOnlyQuantPass();

// Lowers quantize and dequantize ops to a bunch of basic elementwise ops on
// gpu.
std::unique_ptr<OperationPass<func::FuncOp>>
createDiscLowerGpuQuantizeAndDequantizePass();

// Convert mhlo.dynamic_slice to mhlo.real_dynamic_slice
std::unique_ptr<OperationPass<FuncOp>> createDiscDynamicSliceConverterPass();

// Sparse op rewriter
std::unique_ptr<OperationPass<FuncOp>> createDiscSparseOpRewriterPass();

// Inserts dealloc ops for some disc specific ops (e.g. custom_call_v2 op).
std::unique_ptr<OperationPass<FuncOp>> createDiscBufferDeallocationPass();

// Rewrites custom call ops according to its layout attribute.
std::unique_ptr<OperationPass<func::FuncOp>> createDiscCustomCallRewriterPass();

// Convert compute-intensive fusion to the call of FuncOp.
std::unique_ptr<OperationPass<ModuleOp>> createDiscCompIntensFusionToFuncPass();

// Convert the functions representing compute-intensive fusion into CUDA source
// code.
std::unique_ptr<OperationPass<ModuleOp>>
createDiscCompIntensFusionToCUDASourcePass(int cc_major = 8, int cc_minor = 0);

// Compile GPU source code to library, possibly with host side logic.
std::unique_ptr<OperationPass<ModuleOp>> createDiscGPUSourceToLibPass(
    int cc_major = 8, int cc_minor = 0);

// Legalizes transform-based fusion pattern to loop.
std::unique_ptr<OperationPass<func::FuncOp>>
createDiscTransformLegalizeToLoopPass(bool gpuEnabled = false,
                                      const std::string& filename = "",
                                      bool expensiveCheck = false);

// Duplicate and fuse some computation into their fusion consumer to reduce
// memory footprint.
std::unique_ptr<OperationPass<func::FuncOp>>
createDiscDuplicateComputationAfterFusionPass();

}  // namespace disc_ral
}  // namespace mlir

namespace mlir {
namespace mhlo_disc {

// Legalizes mhlo_disc ops to lmhlo_disc ops.
std::unique_ptr<OperationPass<ModuleOp>> createDiscLegalizeToLhloPass();

std::unique_ptr<OperationPass<ModuleOp>> createDiscLhloRewriterPass();

}  // namespace mhlo_disc
}  // namespace mlir

#endif  // DISC_TRANSFORMS_PASSES_H_

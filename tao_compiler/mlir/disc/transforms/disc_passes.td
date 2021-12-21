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

include "mlir/Pass/PassBase.td"

def DiscRalInjectExecutionContextPass : Pass<"disc-inject-execution-context", "ModuleOp"> {
  let summary = "Inject DISC RAL execution context.";
  let constructor = "createDiscInjectExecutionContextPass()";
  let options = [
    Option<"entry_func_name_", "entry-func-name", "std::string",
           /*default=*/"\"main\"", "Name of the entry function.">,
  ];
}

def DiscToLLVMPass : Pass<"disc-to-llvm", "ModuleOp"> {
  let summary = "Pass for applying LLVM legalization patterns.";
  let constructor = "createDiscToLLVMPass()";
}

def SplitLargeOpsPass : FunctionPass<"split-large-ops"> {
  let summary = "Split large hlo ops (e.g. having too many operands) into a "
                "sequence of hlo ops.";
  let constructor = "createDiscSplitLargeOpsPass()";
  let options = [
    Option<"max_num_operands_per_op_", "max-num-operands-per-op", "int",
           /*default=*/"32", "Maximum allowed number of operands per op.">,
  ];
}

def ReviseArgumentsForStaticRankPass : Pass<"disc-tf-revise-args-for-static-rank", "ModuleOp"> {
  let summary = "extract the const/fixed-shape input info from jit static rank compilation cache, and update the info in the IR.";
  let constructor = "createReviseArgsForStaticRankPass()";
}

def DiscLhloLegalizeRootsToParallelLoopsPass : FunctionPass<"disc-lhlo-legalize-roots-to-parallel-loops"> {
  let summary = "Legalize the roots of lmhlo.fusion to parallel loops.";
  let constructor = "createDiscLhloLegalizeRootsToParallelLoopsPass()";
  let dependentDialects = [
    "mlir::scf::SCFDialect",
    "mlir::memref::MemRefDialect",
    "mlir::math::MathDialect",
    "gpu::GPUDialect",
    "disc_shape::DISCShapeDialect"
  ];
}

def ConvRewriterPass : FunctionPass<"disc-conv-rewriter"> {
  let summary = "Canonicalize conv ops to be suitable for lowering to cudnn lib calls.";
  let constructor = "createDiscConvRewriter()";
}

def DotRewriterPass : FunctionPass<"disc-dot-rewriter"> {
  let summary = "Rewrite dot to fold transpose.";
  let constructor = "createDiscDotRewriterPass()";
}

def GpuConvPaddingLegalizationPass : FunctionPass<"disc-gpu-conv-padding-legalization"> {
  let summary = "Convert conv ops' padding value to be suitable for lowering to cudnn lib calls.";
  let constructor = "createDiscGpuConvPaddingLegalization()";
}

def DiscSpecializeFusionWithSpeculationPass : FunctionPass<"disc-specialize-fusion-with-speculation"> {
  let summary = "Pass for specializing a fusion op with some speculations.";
  let constructor = "createDiscSpecializeFusionWithSpeculationPass()";
  let options = [
    Option<"cc_major_", "gpu-sm-cc-major", "int",
            /*default=*/"7", "gpu sm cc_major.">,
    Option<"cc_minor_", "gpu-sm-cc-minor", "int",
            /*default=*/"5", "gpu sm cc_minor.">,
  ];
}

def ElementTypeConverterPass : FunctionPass<"disc-element-type-converter"> {
  let summary = "Eliminates certain element types as the input or output of ops "
                "by inserting Convert ops.";
  let constructor = "createDiscElementTypeConverterPass()";
  let options = [
    Option<"enable_fp16_gemm_", "enable-fp16-gemm", "bool",
           /*default=*/"false", "Enable fp16 for GEMM or not.">,
  ];
}

def MapParallelLoopsPass : FunctionPass<"map-parallel-loops-to-gpu"> {
  let summary = "Greedily maps loops to GPU hardware dimensions.";
  let constructor = "createMapParallelLoopsPass()";
  let description = [{
    Greedily maps loops to GPU hardware dimensions.
  }];
}

def ParallelLoopCollapsing : FunctionPass<"disc-parallel-loop-collapsing"> {
  let summary = "Collapse parallel loops";
  let constructor = "createDiscParallelLoopCollapsingPass()";
}

def SCFParallelLoopTiling : FunctionPass<"disc-parallel-loop-tiling"> {
  let summary = "Tile parallel loops";
  let constructor = "createParallelLoopTilingPass()";
  let options = [
    ListOption<"tileSizes", "parallel-loop-tile-sizes", "int64_t",
               "Factors to tile parallel loops by",
               "llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated">,
    Option<"withInboundCheck", "with-inbound-check", "bool",
           /*default=*/"false",
           "Perform tiling with fixed upperbound with inbound check "
           "for internal loops">
  ];
  let dependentDialects = ["mlir::AffineDialect"];
}

def DiscFusionPass : FunctionPass<"disc-fusion"> {
  let summary = "Fuse lmhlo ops to kLoop/kInput fusion patterns.";
  let constructor = "createDiscFusionPass()";
  let options = [
    Option<"gpu_enabled_", "gpu-enabled", "bool",
           /*default=*/"true", "whether gpu is available.">,
    Option<"fusion_strategy_", "fusion-strategy", "std::string",
           /*default=*/"\"base\"", "Name of the fusion strategy.">,
  ];
}

def DiscMarkShapeCalculationPass : Pass<"disc-mhlo-mark-shape-calc", "ModuleOp"> {
  let summary = "Identify and mark with an attribute operations performing shape calculation";
  let constructor = "createDiscMarkShapeCalcOpPass()";
}

def DiscConstToRALPass : Pass<"disc-const-to-ral", "ModuleOp"> {
  let summary = "convert lhlo ConstOp to ral const.";
  let constructor = "createDiscConstToRALPass()";
  let options = [
    Option<"metadata_file_path_", "metadata-file-name", "std::string",
           /*default=*/"\"metadata.pbtxt\"", "Name of the metadata file.">,
  ];
}

def PlaceOpsPass : Pass<"mhlo-place-ops", "ModuleOp"> {
  let summary = "Place shape calculation Ops on Cpu";
  let constructor = "createPlacerPass()";
  let options = [
    Option<"on_gpu_", "on-gpu", "bool",
           /*default=*/"true", "Data calculation Ops are placed on GPU or not">,
  ];
}

def ReviseGpuKernelOutliningPass : Pass<"disc-revise-gpu-kernel-outlining", "ModuleOp"> {
  let summary = "Revise the kernel outlining by expanding the host memref into scalars";
  let constructor = "createReviseGpuKernelOutliningPass()";
}

def DiscLowerToLibraryCallPass : FunctionPass<"disc-lower-to-library-call"> {
  let summary = "Lower some specific ops to library calls (modeled by `disc_ral.launch` op).";
  let constructor = "createDiscLowerToLibraryCallPass()";
}

def DiscAssignMemorySpacePass : Pass<"disc-assign-memory-space", "ModuleOp"> {
  let summary = "Assign memory space for each memref.";
  let constructor = "createDiscAssignMemorySpacePass()";
  let options = [
    Option<"entry_func_name_", "entry-func-name", "std::string",
           /*default=*/"\"main\"", "Name of the entry function.">,
    Option<"gpu_enabled_", "gpu-enabled", "bool",
           /*default=*/"true", "whether gpu is available.">,
  ];
}

def GpuKernelToBlobPass : Pass<"disc-gpu-kernel-to-blob", "gpu::GPUModuleOp"> {
  let summary = "Pass to annotate GPU Module with its PTX";
  let options = [
    Option<"cc_major_", "gpu-sm-cc-major", "int",
            /*default=*/"7", "gpu sm cc_major.">,
    Option<"cc_minor_", "gpu-sm-cc-minor", "int",
            /*default=*/"5", "gpu sm cc_minor.">,
    Option<"multi_cc_support_", "multi-cc-support", "bool",
            /*default=*/"false", "multiple gpu types support in AOT compilation.">,
    Option<"multi_cc_support_dbg_ptx_only_", "multi-cc-support-dbg-ptx-only", "bool",
            /*default=*/"false", "compile to ptx only, only valid with multi-cc-support.">,
    Option<"blob_annotation_", "blob-annotation", "std::string",
           /*default=*/"\"gpu.binary_blob\"", "Blob attribute name">,
  ];
  let constructor = "CreateDiscGpuKernelToBlobPass()";
}

def ConvertShapeToStandardPass : FunctionPass<"disc-convert-shape-to-std"> {
  let summary = "Pass to convert shape ops to std dialect";
  let constructor = "createDiscConvertShapeToStandardPass()";
}

def ConvertTensorToStandardPass : FunctionPass<"disc-convert-tensor-to-std"> {
  let summary = "Pass to convert tensor ops to std dialect";
  let constructor = "createDiscConvertTensorToStandardPass()";
}

def InputInlineFusionPass : FunctionPass<"disc-input-inline-fusion"> {
  let summary = "Fuse the rest of the lmhlo nodes into the parallel loops after the roots "
    "has been expanded into loops in LhloLegalizeRootsToParallelLoops";
  let constructor = "createDiscInputInlineFusionPass()";
  let dependentDialects = [
    "scf::SCFDialect",
    "memref::MemRefDialect",
    "disc_shape::DISCShapeDialect"
  ];
}

def RemoveShapeConstraintsPass : FunctionPass<"disc-remove-shape-constraint"> {
  let summary = "Pass to remove all shape constraint ops";
  let constructor = "createDiscRemoveShapeConstraintsPass()";
}

def ConvertHloToStandardPass : FunctionPass<"disc-convert-hlo-to-std"> {
  let summary = "Pass to convert some hlo ops to std dialect";
  let constructor = "createDiscConvertHloToStandardPass()";
}

def StdBufferizePass : FunctionPass<"disc-std-bufferize"> {
  let summary = "Pass to convert tensor std.constant ops to hlo dialect";
  let constructor = "createDiscStdBufferizePass()";
}

def DiscMemRefCSEPass : FunctionPass<"disc-memref-cse"> {
  let summary = "Perform CSE on memref.load";
  let constructor = "createDiscMemRefCSEPass()";
}

def DiscLowerTfPass : FunctionPass<"disc-lower-tf"> {
  let summary = "Lowering some tensorflow ops";
  let constructor = "createDiscLowerTfPass()";
}

def RemoveDeadBufferPass : FunctionPass<"disc-remove-dead-buffer"> {
  let summary = "Pass to remove buffers that are not accessed by others";
  let constructor = "createDiscRemoveDeadBufferPass()";
}

def AssignKernelNamePass : Pass<"disc-assign-kernel-name", "ModuleOp"> {
  let summary = "Assign a meaningful name for each gpu kernel.";
  let constructor = "createDiscAssignKernelNamePass()";
}

def DiscLowerGpuOpsToNVVMOpsPass : Pass<"disc-convert-gpu-to-nvvm", "gpu::GPUModuleOp"> {
  let summary = "Generate NVVM operations for gpu operations";
  let constructor = "createDiscLowerGpuOpsToNVVMOpsPass()";
  let dependentDialects = ["NVVM::NVVMDialect", "memref::MemRefDialect"];
  let options = [
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">
  ];
}

def DiscLowerGpuOpsToROCDLOpsPass : Pass<"disc-convert-gpu-to-rocdl", "gpu::GPUModuleOp"> {
  let summary = "Generate ROCDL operations for gpu operations";
  let constructor = "createDiscLowerGpuOpsToROCDLOpsPass()";
  let dependentDialects = ["ROCDL::ROCDLDialect", "memref::MemRefDialect"];
  let options = [
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">
  ];
}

def UnhandledAtomicRMWConverterPass : FunctionPass<"disc-convert-unhandled-atomic-rmw"> {
  let summary = "Convert unhandled AtomicRMWOp to GenericAtomicRMWOp.";
  let constructor = "createDiscUnhandledAtomicRMWConverterPass()";
  let dependentDialects = ["memref::MemRefDialect"];
}

def DiscShapeSimplifierPass : Pass<"disc-shape-simplifier", "ModuleOp"> {
  let summary = "propagate some known shape information.";
  let constructor = "createDiscShapeSimplifierPass()";
  let dependentDialects = ["disc_shape::DISCShapeDialect"];
  let options = [
    Option<"entry_func_name_", "entry-func-name", "std::string",
           /*default=*/"\"main\"", "Name of the entry function.">,
    Option<"insert_tie_shape_", "insert-tie-shape", "bool",
           /*default=*/"false", "whether tie shape op is inserted.">,
  ];
}

def DiscMathApproximationPass : FunctionPass<"disc-math-approximation"> {
  let summary = "using approximation for some special math ops.";
  let constructor = "createDiscMathApproximationPass()";
  let dependentDialects = [
    "mlir::math::MathDialect",
    "mlir::LLVM::LLVMDialect"
  ];
}

def DiscFlattenMemrefAccessPass : FunctionPass<"disc-flatten-memref-access"> {
  let summary = "flatten memref access to its 1D format.";
  let constructor = "createDiscFlattenMemrefAccessPass()";
  let dependentDialects = [
    "mlir::disc_shape::DISCShapeDialect"
  ];
}

def Canonicalizer : Pass<"disc-canonicalize"> {
  let summary = "Canonicalize operations";
  let description = [{
    This pass performs various types of canonicalizations over a set of
    operations. See [Operation Canonicalization](Canonicalization.md) for more
    details.
  }];
  let constructor = "createDiscCanonicalizerPass()";
  let options = [
    // These two options provide filtering for which patterns are applied. These
    // should be passed directly to the FrozenRewritePatternSet when it is
    // created.
    ListOption<"disabledPatterns_", "disable-patterns", "std::string",
               "Labels of patterns that should be filtered out during"
               " application",
               "llvm::cl::MiscFlags::CommaSeparated">,
    ListOption<"enabledPatterns_", "enable-patterns", "std::string",
               "Labels of patterns that should be used during"
               " application, all other patterns are filtered out",
               "llvm::cl::MiscFlags::CommaSeparated">
  ];
}

def DiscMemrefCanonicalizer : FunctionPass<"disc-memref-canonicalize"> {
  let summary = "Do some memref-related cleanup.";
  let constructor = "createDiscMemrefCanonicalizerPass()";
}

def DiscOutlineCpuKernel : Pass<"disc-outline-cpu-kernel", "ModuleOp"> {
  let summary = "outline each cpu kernel to a dedicated function.";
  let constructor = "createDiscOutlineCpuKernelPass()";
}

def DiscCpuMapParallelLoop : FunctionPass<"disc-cpu-map-parallel-loop"> {
  let summary = "assign a parallel schedule for each parallel op on cpu.";
  let constructor = "createDiscCpuMapParallelLoopPass()";
}

def DiscStitchFusion : FunctionPass<"disc-stitch-fusion"> {
  let summary = "Lowring stitch fusion pattern to parallel for loops.";
  let constructor = "createDiscStitchFusionPass()";
}

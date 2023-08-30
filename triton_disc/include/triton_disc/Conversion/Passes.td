#ifndef TRITON_DISC_CONVERSION_PASSES
#define TRITON_DISC_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonGPUToMLIRGPU: Pass<"convert-tritongpu-to-mlir-gpu", "mlir::ModuleOp"> {
    let summary = "Convert TritonGPU to Parallel Loop";
    let description = [{

    }];
    let constructor = "mlir::triton_disc::createTritonGPUToMLIRGPUPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::tensor::TensorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect"];

}

#endif

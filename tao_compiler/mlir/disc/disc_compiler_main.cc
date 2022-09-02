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

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/lhlo/transforms/register_passes.h"
#include "mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/register_passes.h"
#include "mlir/ExecutionEngine/OptUtils.h"  // from @llvm-project
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"          // from @llvm-project
#include "mlir/Pass/PassManager.h"       // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Export.h"  // from @llvm-project
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "stablehlo/dialect/ChloOps.h"
#include "tensorflow/compiler/mlir/disc/IR/disc_ral_ops.h"
#include "tensorflow/compiler/mlir/disc/IR/hlo_disc_ops.h"
#include "tensorflow/compiler/mlir/disc/IR/lhlo_disc_ops.h"
#include "tensorflow/compiler/mlir/disc/disc_compiler.h"
#include "tensorflow/compiler/mlir/disc/transforms/register_passes.h"
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/xla/stream_executor/lib/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"

#ifndef TAO_CPU_ONLY
#if TENSORFLOW_USE_ROCM
#define GPU_SUCCESS hipSuccess
#include "tensorflow/stream_executor/rocm/rocm_driver_wrapper.h"
#else
#define GPU_SUCCESS CUDA_SUCCESS
#include "cuda.h"
#endif
#endif

namespace mlir {
namespace disc_ral {
namespace {

llvm::cl::opt<std::string> inputFilename{
    llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::init("-")};
llvm::cl::opt<std::string> outputFilename{llvm::cl::Positional,
                                          llvm::cl::desc("<output file>"),
                                          llvm::cl::init("out.so")};
llvm::cl::opt<bool> MultiCCSupport(
    "multi-cc-support",
    llvm::cl::desc("Enable multiple gpu card type support during compilation"),
    llvm::cl::init(false));
llvm::cl::opt<bool> MultiCCSupportDbgPtxOnly(
    "multi-cc-support-dbg-ptx-only",
    llvm::cl::desc(
        "Compile to PTX only, only valid with multi-cc-support is true"),
    llvm::cl::init(false));

static mlir::OwningOpRef<mlir::ModuleOp> parseMLIRInput(StringRef inputFilename,
                                                        MLIRContext* context) {
  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  return mlir::OwningOpRef<mlir::ModuleOp>(
      parseSourceFile<mlir::ModuleOp>(sourceMgr, context));
}

#ifndef TAO_CPU_ONLY

template <typename Error>
void dumpGpuError(Error error) {
#if defined(TENSORFLOW_USE_ROCM)
  llvm::errs() << "HIP Error: " << hipGetErrorString(error) << "\n";
#elif defined(GOOGLE_CUDA)
  const char* errstr;
  cuGetErrorString(error, &errstr);
  llvm::errs() << "CUDA error: " << errstr << "\n";
#endif
}

#define RETURN_ON_GPU_ERROR(expr, msg) \
  {                                    \
    auto _gpu_error = (expr);          \
    if (_gpu_error != GPU_SUCCESS) {   \
      llvm::errs() << msg << "\n";     \
      dumpGpuError(_gpu_error);        \
      return 1;                        \
    }                                  \
  }

#if TENSORFLOW_USE_ROCM

int InitGPU(GpuDeviceInfo& ctx) {
  RETURN_ON_GPU_ERROR(hipInit(0), "hipInit");
  // TODO: cc is not used for DCU for now
  ctx.cc_major = 0;
  ctx.cc_minor = 0;
  return 0;
}

#else

int InitGPU(GpuDeviceInfo& ctx) {
  CUdevice device;
  CUcontext context;
  RETURN_ON_GPU_ERROR(cuInit(0), "cuInit");
  RETURN_ON_GPU_ERROR(cuDeviceGet(&device, /*device_ordinal*/ 0),
                      "cuDeviceGet");
  RETURN_ON_GPU_ERROR(cuCtxCreate(&context, 0, device), "cuCtxCreate");
  RETURN_ON_GPU_ERROR(
      cuDeviceComputeCapability(&ctx.cc_major, &ctx.cc_minor, device),
      "cuDeviceComputeCapability");
  RETURN_ON_GPU_ERROR(
      cuDeviceGetAttribute(&ctx.sm_count,
                           CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device),
      "cuDeviceGetAttribute (MULTIPROCESSOR_COUNT)");
  RETURN_ON_GPU_ERROR(
      cuDeviceGetAttribute(&ctx.max_threads_per_sm,
                           CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                           device),
      "cuDeviceGetAttribute (MAX_THREADS_PER_MULTIPROCESSOR)");
  return 0;
}

#endif
#endif

int RealMain() {
  mlir::registerAllPasses();
  mlir::registerTensorFlowPasses();
  mlir::mhlo::registerAllMhloPasses();
  mlir::lmhlo::registerAllLmhloPasses();
  mlir::disc_ral::registerAllDiscPasses();
  mlir::mhlo_disc::registerAllMhloDiscPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  registry.insert<mlir::mhlo::MhloDialect>();
  registry.insert<mlir::mhlo_disc::MhloDiscDialect>();
  registry.insert<mlir::chlo::ChloDialect>();
  registry.insert<mlir::lmhlo::LmhloDialect>();
  registry.insert<mlir::lmhlo_disc::LmhloDiscDialect>();
  registry.insert<mlir::lmhlo_gpu::LmhloGpuDialect>();
  registry.insert<mlir::disc_ral::RalDialect>();
  registry.insert<mlir::TF::TensorFlowDialect>();

  MLIRContext context(registry);
  auto m = parseMLIRInput(inputFilename, &context);
  if (!m) {
    llvm::errs() << "could not parse the input file\n";
    return 1;
  }

  ModuleOp module = m.get();

  if (VLOG_IS_ON(0)) {
    llvm::dbgs() << "======== BEGIN Original Module =========\n";
    module.dump();
    llvm::dbgs() << "\n======= END Original Module ==========\n";
  }

  llvm::dbgs() << "[[ INFO ]] Running TF2XLA\n";
  auto s = tensorflow::ConvertTF2MlirHlo(module);
  if (!s.ok()) {
    llvm::dbgs() << "ConvertTF2MlirHlo failed: " << s.ToString() << "\n";
  }

  if (VLOG_IS_ON(0)) {
    llvm::dbgs() << "======== BEGIN After TF2HLO =========\n";
    module.dump();
    llvm::dbgs() << "\n======= END After TF2HLO ==========\n";
  }

  DISCLoweringOptions disc_options(outputFilename);
#ifndef TAO_CPU_ONLY
  disc_options.gpu_options.multi_cc_support = MultiCCSupport;
  disc_options.gpu_options.multi_cc_support_dbg_ptx_only =
      MultiCCSupportDbgPtxOnly;
  if (InitGPU(disc_options.gpu_info)) {
    return 1;
  };
#else
  disc_options.mode = mlir::disc_ral::CodeGenMode::kCpuCentric;
#endif

  if (failed(LowerHLOToSharedLibrary(module, disc_options))) {
    llvm::errs() << "could not convert hlo to shared lib file\n";
    return 1;
  }

  return 0;
}

}  // namespace
}  // namespace disc_ral
}  // namespace mlir

int main(int argc, char** argv) {
  tensorflow::InitMlir y(&argc, &argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::registerPassManagerCLOptions();
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "DISC Compiler\n");

  return mlir::disc_ral::RealMain();
}

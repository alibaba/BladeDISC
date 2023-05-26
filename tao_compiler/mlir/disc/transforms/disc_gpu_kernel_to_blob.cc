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

#include <fstream>

#include "absl/strings/str_cat.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/codegen_utils.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_asm_opts_util.h"
#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/utils.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/gpu/target_constants.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_type_conversion_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/tsl/platform/cuda_libdevice_path.h"

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
#include "tensorflow/compiler/xla/stream_executor/gpu/asm_compiler.h"
#endif
#if (TENSORFLOW_USE_ROCM || TENSORFLOW_USE_DCU)
#include "rocm/rocm_config.h"
#include "tensorflow/compiler/xla/stream_executor/rocm/rocm_driver_wrapper.h"
#include "tensorflow/core/platform/rocm_rocdl_path.h"
#define CUDA_SUCCESS hipSuccess
#define ROCM_CALL(func)                                             \
  {                                                                 \
    hipError_t e = (func);                                          \
    CHECK(e == hipSuccess) << "ROCM HIP: " << hipGetErrorString(e); \
  }
#endif

namespace mlir {
namespace disc_ral {
namespace {

#if TENSORFLOW_USE_ROCM
static std::string RocmCurrentArch() {
  int device_id;
  ROCM_CALL(hipGetDevice(&device_id));
  hipDeviceProp_t prop;
  ROCM_CALL(hipGetDeviceProperties(&prop, device_id));
  auto arch_type = prop.gcnArch;
  std::string arch_str = "gfx" + std::to_string(arch_type);
  if (arch_str == "gfx910") {
    arch_str = "gfx90a";
  }
  return std::move(arch_str);
}
#endif

using xla::InternalError;
using xla::llvm_ir::AsArrayRef;
using xla::llvm_ir::AsStringRef;

class GpuKernelToBlobPass
    : public GpuKernelToBlobPassBase<GpuKernelToBlobPass> {
 public:
  GpuKernelToBlobPass(int cc_major, int cc_minor, bool multi_cc_support,
                      bool multi_cc_support_dbg_ptx_only,
                      StringRef blob_annotation) {
    blob_annotation_ = blob_annotation.str();
    cc_major_ = cc_major;
    cc_minor_ = cc_minor;
    multi_cc_support_ = multi_cc_support;
    multi_cc_support_dbg_ptx_only_ = multi_cc_support_dbg_ptx_only;
    if (multi_cc_support) {
      // TODO: support customized fusion speculation for multiple GPU
      // generations with exact sm count. Note that we cannot distinguish GPUs
      // with only cc numbers (e.g., A10 and A40 are both sm_86).
      LOG(INFO) << "Multiple GPU compute capability compilation is enabled. "
                   "The AOT compiled binary is functional on sm_60, sm_70, "
                   "sm_75, sm_80 and sm_86. While the optimal performance can "
                   "be achived only on currently GPU generation that BladeDISC "
                   "executes on.";
    }
  }

  void runOnOperation() override {
    mlir::gpu::GPUModuleOp gpu_module = getOperation();
    if (multi_cc_support_) {
      VLOG(2) << "Multi compute capability support";
      for (auto item : c_MULTI_SM_CONFIG) {
        const std::string& name = item.first;
        if (multi_cc_support_dbg_ptx_only_) {
          if (name.find("compute") == std::string::npos) continue;
          VLOG(2) << "Multi compute capability support with PTX only";
        }

        auto blob_or = GetGpuBinaryBlob(gpu_module, std::get<0>(item.second),
                                        std::get<1>(item.second),
                                        std::get<2>(item.second));
        if (!blob_or.ok()) {
          gpu_module.emitError(blob_or.status().error_message());
          return signalPassFailure();
        }
        const auto& blob = blob_or.value();
        std::string blob_string(blob.begin(), blob.end());
        std::string attr_str = std::string(kGpuBinaryAttrName) + "_" + name;
        gpu_module->setAttr(attr_str,
                            mlir::StringAttr::get(&getContext(), blob_string));
      }
    } else {
      VLOG(2) << "JIT mode";
      auto blob_or = GetGpuBinaryBlob(gpu_module, cc_major_, cc_minor_);
      if (!blob_or.ok()) {
        gpu_module.emitError(blob_or.status().error_message());
        return signalPassFailure();
      }
      const auto& blob = blob_or.value();
      std::string blob_string(blob.begin(), blob.end());
      gpu_module->setAttr(blob_annotation_,
                          mlir::StringAttr::get(&getContext(), blob_string));
    }
  }

  static xla::Status ExecuteProgram(const std::string& program,
                                    const std::vector<llvm::StringRef>& args) {
    std::string error_message;
    int result = llvm::sys::ExecuteAndWait(
        program, AsArrayRef(args), llvm::None, {}, 0, 0, &error_message);
    if (result) {
      return xla::InternalError("llc execute fail: %s, error code %d",
                                error_message, result);
    }
    return xla::OkStatus();
  }

  xla::StatusOr<std::vector<uint8_t>> GetGpuBinaryBlob(
      mlir::gpu::GPUModuleOp gpu_module, int cc_major, int cc_minor,
      bool virtual_compute_arch = false) {
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(gpu_module, llvmContext);

#if TENSORFLOW_USE_ROCM
#if TENSORFLOW_USE_ROCM_COMPILE_TOOLKIT
    std::string libdevice_dir = tensorflow::RocdlRoot();
    std::string rocm_path;
    tensorflow::ReadStringFromEnvVar("DISC_ROCM_PATH", "/opt/rocm", &rocm_path);
#if (TF_ROCM_VERSION >= 30900 || TENSORFLOW_USE_DCU)
    libdevice_dir = tensorflow::io::JoinPath(rocm_path, "amdgcn/bitcode");
#else
    libdevice_dir = tensorflow::io::JoinPath(rocm_path, "lib");
#endif
    auto arch_str = RocmCurrentArch();
    TF_RETURN_IF_ERROR(xla::gpu::LinkWithBitcodeVector(
        llvmModule.get(), xla::gpu::GetROCDLPaths(arch_str, libdevice_dir)));

    std::string llvm_path_1;
    tensorflow::ReadStringFromEnvVar("DISC_ROCM_BACKEND_PATH",
                                     "/opt/rocm/llvm/bin/", &llvm_path_1);
    std::string llvm_path_2 = tensorflow::io::JoinPath("/opt/rocm", "hcc/bin");
    std::string llvm_path_3 = tensorflow::io::JoinPath("/opt/rocm", "llvm/bin");
    auto llc_program = llvm::sys::findProgramByName(
        "llc", {llvm_path_1, llvm_path_2, llvm_path_3});
    auto lld_program = llvm::sys::findProgramByName(
        "ld.lld", {llvm_path_1, llvm_path_2, llvm_path_3});
    if (!llc_program || !lld_program) {
      return xla::InternalError("unable to find llc or ld.lld in PATH: %s, %s",
                                llc_program.getError().message(),
                                lld_program.getError().message());
    }
    VLOG(2) << "llc found in path: " << *llc_program
            << ", ld.lld found in path: " << *lld_program;

    std::string random_number = std::to_string(tensorflow::random::New64());
    std::string tmp_path = "/tmp/";
    std::string ll_path = tmp_path + random_number + ".ll";
    std::string isabin_path = tmp_path + random_number + ".o";
    std::string hsaco_path = tmp_path + random_number + ".hsaco";

    std::error_code ec;
    std::unique_ptr<llvm::raw_fd_ostream> ll_fs(
        new llvm::raw_fd_ostream(ll_path, ec, llvm::sys::fs::OF_None));
    llvmModule->print(*ll_fs, nullptr);
    ll_fs->flush();

    std::string opt_level;
    tensorflow::ReadStringFromEnvVar("DISC_ROCM_BACKEND_OPT_LEVEL", "3",
                                     &opt_level);
    if (VLOG_IS_ON(2)) {
      // Dump asm file
      std::string asm_path = tmp_path + random_number + ".asm";
      std::vector<llvm::StringRef> llc_args{
          AsStringRef("llc"),
          AsStringRef("-O"),
          AsStringRef(opt_level),
          AsStringRef("-mtriple"),
          AsStringRef("amdgcn-amd-amdhsa"),
          AsStringRef("-mcpu"),
          AsStringRef(arch_str),
          AsStringRef("-amdhsa-code-object-version"),
          AsStringRef("3"),
          AsStringRef("-filetype"),
          AsStringRef("asm"),
          AsStringRef("-o"),
          AsStringRef(asm_path),
          AsStringRef(ll_path)};
      TF_RETURN_IF_ERROR(ExecuteProgram(*llc_program, llc_args));
      VLOG(0) << "Asm file is dumped to: " << asm_path;
    }
    std::vector<llvm::StringRef> llc_args{
        AsStringRef("llc"),
        AsStringRef("-O"),
        AsStringRef(opt_level),
        AsStringRef("-mtriple"),
        AsStringRef("amdgcn-amd-amdhsa"),
        AsStringRef("-mcpu"),
        AsStringRef(arch_str),
        AsStringRef("-amdhsa-code-object-version"),
        AsStringRef("3"),
        AsStringRef("-filetype"),
        AsStringRef("obj"),
        AsStringRef("-o"),
        AsStringRef(isabin_path),
        AsStringRef(ll_path)};
    TF_RETURN_IF_ERROR(ExecuteProgram(*llc_program, llc_args));

    std::vector<llvm::StringRef> lld_args{
        AsStringRef("ld.lld"),  AsStringRef("-flavor"),   AsStringRef("gnu"),
        AsStringRef("-shared"), AsStringRef(isabin_path), AsStringRef("-o"),
        AsStringRef(hsaco_path)};
    TF_RETURN_IF_ERROR(ExecuteProgram(*lld_program, lld_args));

    // Read HSACO.
    std::ifstream hsaco_file(hsaco_path, std::ios::binary | std::ios::ate);
    std::ifstream::pos_type hsaco_file_size = hsaco_file.tellg();

    std::vector<uint8_t> hsaco(hsaco_file_size);
    hsaco_file.seekg(0, std::ios::beg);
    hsaco_file.read(reinterpret_cast<char*>(&hsaco[0]), hsaco_file_size);
    hsaco_file.close();
    bool keep_tempfiles = false;
    TF_CHECK_OK(tensorflow::ReadBoolFromEnvVar("DISC_ROCM_KEEP_TEMPFILES",
                                               /*default_val=*/false,
                                               &keep_tempfiles));

    if (!keep_tempfiles) {
      remove(ll_path.c_str());
      remove(isabin_path.c_str());
      remove(hsaco_path.c_str());
    }
    return hsaco;

#else
    xla::HloModuleConfig config;
    xla::DebugOptions options = xla::GetDebugOptionsFromFlags();
    config.set_debug_options(options);

    using AmdGpuHsaco = std::vector<tensorflow::uint8>;
    auto arch_str = RocmCurrentArch();
    // Parse ROCm architecture.
    absl::string_view consumable_arch(arch_str);
    if (!absl::ConsumePrefix(&consumable_arch, "gfx")) {
      return tsl::errors::Internal(
          "Could not parse ROCm architecture prefix (expected gfx)");
    }
    std::string libdevice_dir = tensorflow::RocdlRoot();
    std::string rocm_path;
    tensorflow::ReadStringFromEnvVar("DISC_ROCM_PATH", "/opt/rocm", &rocm_path);

#if (TF_ROCM_VERSION >= 30900 || TENSORFLOW_USE_DCU)
    libdevice_dir = tensorflow::io::JoinPath(rocm_path, "amdgcn/bitcode");
#else
    libdevice_dir = tensorflow::io::JoinPath(rocm_path, "lib");
#endif

    auto llvm_module_copy = llvm::CloneModule(*llvmModule);
    auto kname = gpu_module.getName().str();
    auto hsaco_or = xla::gpu::amdgpu::CompileToHsaco(
        llvm_module_copy.get(), tensorflow::se::RocmComputeCapability{arch_str},
        config, libdevice_dir);

    bool dump_files;
    tensorflow::ReadBoolFromEnvVar("DISC_ROCM_DUMP_FILES", false, &dump_files);

    if (!hsaco_or.ok()) {
      LOG(WARNING) << "LLVM Backend compile HSACO fail.";
    } else if (dump_files) {
      std::error_code ec;
      std::string ll_path = kname + "_debugllvm.ll";
      std::unique_ptr<llvm::raw_fd_ostream> ll_fs(
          new llvm::raw_fd_ostream(ll_path, ec, llvm::sys::fs::OF_None));
      llvmModule->print(*ll_fs, nullptr);
      ll_fs->flush();

      const auto& blob = hsaco_or.value();
      std::string llvmpath = kname + "_debugllvm.hsaco";
      auto myfile = std::fstream(llvmpath, std::ios::out | std::ios::binary);
      myfile.write((char*)blob.data(), blob.size());
      myfile.close();
      LOG(INFO) << "llvm hsaco to " << llvmpath;
      hipModule_t hipmodule;

      hipError_t res = stream_executor::wrap::hipModuleLoadData(
          &hipmodule, reinterpret_cast<const void*>(blob.data()));
      if (res != hipSuccess) {
        LOG(WARNING) << "error for hsoco build " << res << " " << kname;
      } else {
        LOG(INFO) << "Finish for hsoco build " << res << " " << kname;
      }
    } else {
      LOG(WARNING) << "LLVM Backend compile HSACO fail.";
    }
    return hsaco_or;

#endif
#elif GOOGLE_CUDA
    if (!llvmModule) {
      return InternalError("Could not translate MLIR module to NVVM");
    }

    llvmModule->setModuleIdentifier("acme");
    llvmModule->setDataLayout(xla::gpu::nvptx::DataLayout());
    llvmModule->setTargetTriple(xla::gpu::nvptx::TargetTriple());

    xla::HloModuleConfig config;
    xla::DebugOptions options = xla::GetDebugOptionsFromFlags();
    int64_t fast_math_level = 1;
    tensorflow::ReadInt64FromEnvVar("DISC_CUDA_FAST_MATH_LEVEL", 1,
                                    &fast_math_level);
    switch (fast_math_level) {
      case 0:
        // No fast-math at all.
        (*options.mutable_xla_backend_extra_options())["-nvptx-prec-divf32"] =
            "2";
        (*options.mutable_xla_backend_extra_options())["-nvptx-prec-sqrtf32"] =
            "1";
        options.set_xla_gpu_ftz(false);
        break;
      case 1:
        // This is the default value, which is the same with XLA. Nothing will
        // be changed. Note that it sets nvptx-prec-divf32 to 1.
        break;
      case 2:
        // Note that this is not the full fase-math for CUDA. The prec-div is
        // set to 1 instead of 0 for better precision.
        (*options.mutable_xla_backend_extra_options())["-nvptx-prec-divf32"] =
            "1";
        (*options.mutable_xla_backend_extra_options())["-nvptx-prec-sqrtf32"] =
            "0";
        options.set_xla_gpu_ftz(true);
        break;
      case 3:
        // Full fase-math for CUDA.
        (*options.mutable_xla_backend_extra_options())["-nvptx-prec-divf32"] =
            "0";
        (*options.mutable_xla_backend_extra_options())["-nvptx-prec-sqrtf32"] =
            "0";
        options.set_xla_gpu_ftz(true);
        break;
      default:
        break;
    }
    config.set_debug_options(options);

    auto enable_fusion = [](llvm::TargetMachine* target) {
      target->Options.AllowFPOpFusion = llvm::FPOpFusion::FPOpFusionMode::Fast;
    };

    // Compile and collect requested fatbin and PTX images.
    std::vector<tensorflow::se::CubinOrPTXImage> images;
    auto gpu_asm_opts =
        xla::gpu::PtxOptsFromDebugOptions(config.debug_options());

    // [GPU] Fold finding libdevice_dir into PTX compilation
    TF_ASSIGN_OR_RETURN(
        std::string ptx,
        xla::gpu::nvptx::CompileToPtx(
            llvmModule.get(),
            tensorflow::se::CudaComputeCapability{cc_major, cc_minor}, config,
            enable_fusion));

    VLOG(1) << "PTX code: \n" << ptx;

    std::vector<uint8_t> gpu_asm;
    if (virtual_compute_arch) {
      std::vector<uint8_t> ptx_bytes;
      std::copy(ptx.begin(), ptx.end(), std::back_inserter(ptx_bytes));
      tensorflow::se::CubinOrPTXImage image{
          absl::StrCat("compute_", cc_major, cc_minor), std::move(ptx_bytes)};
      TF_ASSIGN_OR_RETURN(gpu_asm,
                          tensorflow::se::BundleGpuAsm({image}, gpu_asm_opts));
    } else {
      TF_ASSIGN_OR_RETURN(std::vector<uint8_t> cubin,
                          tensorflow::se::CompileGpuAsm(
                              cc_major, cc_minor, ptx.c_str(), gpu_asm_opts));
      tensorflow::se::CubinOrPTXImage image{
          absl::StrCat("sm_", cc_major, cc_minor), std::move(cubin)};
      TF_ASSIGN_OR_RETURN(gpu_asm,
                          tensorflow::se::BundleGpuAsm({image}, gpu_asm_opts));
    }

    bool keep_tempfiles = false;
    TF_CHECK_OK(tensorflow::ReadBoolFromEnvVar("DISC_CUDA_KEEP_TEMPFILES",
                                               /*default_val=*/false,
                                               &keep_tempfiles));
    if (!keep_tempfiles) {
      auto* env = tsl::Env::Default();
      std::vector<std::string> tempdir_vector;
      env->GetLocalTempDirectories(&tempdir_vector);
      if (tempdir_vector.empty()) {
        return xla::InternalError(
            "Unable to locate a temporary directory for compile-time "
            "artifacts.");
      }
      std::string tempdir_name = tempdir_vector.front();
      VLOG(1) << "Compile-time artifacts located at: " << tempdir_name;
      std::string random_number = std::to_string(tensorflow::random::New64());
      std::string filename =
          absl::StrCat(llvmModule->getModuleIdentifier(), random_number);
      std::string ir_path =
          tensorflow::io::JoinPath(tempdir_name, absl::StrCat(filename, ".ll"));
      std::string ptx_path = tensorflow::io::JoinPath(
          tempdir_name, absl::StrCat(filename, ".ptx"));
      std::string cubin_path = tensorflow::io::JoinPath(
          tempdir_name, absl::StrCat(filename, ".cubin"));

      // Dump LLVM IR.
      std::error_code ec;
      std::unique_ptr<llvm::raw_fd_ostream> ir_fs(
          new llvm::raw_fd_ostream(ir_path, ec, llvm::sys::fs::OF_None));
      llvmModule->print(*ir_fs, nullptr);
      ir_fs->flush();

      // Dump PTX.
      auto status = tsl::WriteStringToFile(env, ptx_path, ptx);
      if (!status.ok()) {
        LOG(ERROR) << "Could not write PTX to " << ptx_path << ": " << status;
      }

      // Dump cubin.
      std::ofstream fout(cubin_path, std::ios::binary);
      fout.write(reinterpret_cast<const char*>(gpu_asm.data()),
                 gpu_asm.size() * sizeof(uint8_t));
      fout.close();
    }
    return gpu_asm;

#endif

    return InternalError(
        "Neither TENSORFLOW_USE_ROCM nor GOOGLE_CUDA are defined."
        " Did you specify either --config=rocm or --config=cuda ?");
  }

 protected:
  void getDependentDialects(DialectRegistry& registry) const override {
    registerLLVMDialectTranslation(registry);
    registerNVVMDialectTranslation(registry);
    OperationPass<gpu::GPUModuleOp>::getDependentDialects(registry);
  }
};

}  // namespace

std::unique_ptr<OperationPass<gpu::GPUModuleOp>> CreateDiscGpuKernelToBlobPass(
    int cc_major, int cc_minor, bool multi_cc_support,
    bool multi_cc_support_dbg_ptx_only, mlir::StringRef blob_annotation) {
  return std::make_unique<GpuKernelToBlobPass>(
      cc_major, cc_minor, multi_cc_support, multi_cc_support_dbg_ptx_only,
      blob_annotation);
}

}  // namespace disc_ral
}  // namespace mlir

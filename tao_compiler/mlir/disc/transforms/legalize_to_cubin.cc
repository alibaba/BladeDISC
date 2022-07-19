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

//===- ConvertKernelFuncToCubin.cpp - MLIR GPU lowering passes ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert gpu kernel functions into a
// corresponding binary blob that can be executed on a CUDA GPU. Currently
// only translates the function itself but no dependencies.
//
//===----------------------------------------------------------------------===//

#include <string>

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "mlir/Conversion/GPUToCUDA/GPUToCUDAPass.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/NVVMIR.h"
#include "tensorflow/compiler/xla/legacy_flags/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/nvptx_backend_lib.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/cuda_libdevice_path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/command_line_flags.h"

// debug only
#include <fstream>
#include <iostream>
#include <streambuf>

#include "llvm/Bitcode/BitcodeWriter.h"
#include "tensorflow/core/platform/env.h"

using mlir::Location;
using mlir::StringRef;
using xla::HloModuleConfig;

using OwnedCubin = std::unique_ptr<std::vector<char>>;
using CubinGenerator =
    std::function<OwnedCubin(const std::string&, Location, StringRef)>;

///* static */ const char* kTargetTriple = "nvptx64-nvidia-gpulib";
/* static */ const char* kTargetTriple = "nvptx64-nvidia-cuda";
/* static */ const char* kDataLayout =
    "e-i64:64-i128:128-v16:16-v32:32-n16:32:64";

namespace mlir {
namespace disc_ral {

namespace {
// TODO(herhut): Move to shared location.
static constexpr const char* kCubinAnnotation = "nvvm.cubin";

std::string DumpModuleToString(const llvm::Module& module) {
  std::string buffer_string;
  llvm::raw_string_ostream ostream(buffer_string);
  module.print(ostream, nullptr);
  ostream.flush();
  return buffer_string;
}

static std::vector<std::string> CandidateCudaRoots(
    const HloModuleConfig& config) {
  return tensorflow::CandidateCudaRoots(
      config.debug_options().xla_gpu_cuda_data_dir());
}

std::string GetLibdeviceDir(const HloModuleConfig& hlo_module_config) {
  std::vector<std::string> candidate_cuda_roots =
      CandidateCudaRoots(hlo_module_config);
  candidate_cuda_roots.push_back(tensorflow::CudaRoot());
  for (const std::string& cuda_root : candidate_cuda_roots) {
    std::string libdevice_dir =
        tensorflow::io::JoinPath(cuda_root, "nvvm", "libdevice");
    VLOG(2) << "Looking for libdevice at " << libdevice_dir;
    if (tensorflow::Env::Default()->IsDirectory(libdevice_dir).ok()) {
      VLOG(2) << "Found libdevice dir " << libdevice_dir;
      return libdevice_dir;
    }
  }
  // GetCudaRootCandidates always includes ".", but but if everything fails, we
  // return it anyway.  Better than returning the empty string.
  return ".";
}

/// A pass converting tagged kernel modules to cubin blobs.
///
/// If tagged as a kernel module, each contained function is translated to NVVM
/// IR and further to PTX. A user provided CubinGenerator compiles the PTX to
/// GPU binary code, which is then attached as an attribute to the function. The
/// function body is erased.
class GpuKernelToCubinPassV2
    : public OperationPass<GpuKernelToCubinPassV2, gpu::GPUModuleOp> {
 public:
  GpuKernelToCubinPassV2(
      CubinGenerator cubinGenerator = compilePtxToCubinForTesting,
      int cc_major = 7, int cc_minor = 5)
      : cubinGenerator(cubinGenerator),
        cc_major_(cc_major),
        cc_minor_(cc_minor) {}

  void runOnOperation() override {
    gpu::GPUModuleOp module = getOperation();

    auto llvmModule = translateModuleToNVVMIR(module);
    if (!llvmModule) return signalPassFailure();

    // Set the target triple and the data layout.
    llvmModule->setTargetTriple(kTargetTriple);
    llvmModule->setDataLayout(kDataLayout);

    HloModuleConfig hlo_module_config;
    hlo_module_config.set_debug_options(
        xla::legacy_flags::DefaultDebugOptionsIgnoringFlags());
    std::string ptx;

    ptx = xla::gpu::CompileToPtx(
              llvmModule.get(), std::make_pair(cc_major_, cc_minor_),
              hlo_module_config, GetLibdeviceDir(hlo_module_config))
              .ValueOrDie();

    // Translate the module to CUBIN and attach the result as attribute to the
    // module.
    if (auto cubinAttr = translateGPUModuleToCubinAnnotation(
            ptx, module.getLoc(), module.getName()))
      module.setAttr(kCubinAnnotation, cubinAttr);
    else
      signalPassFailure();
  }

 private:
  static OwnedCubin compilePtxToCubinForTesting(const std::string& ptx,
                                                Location, StringRef);

  /// Converts llvmModule to cubin using the user-provided generator. Location
  /// is used for error reporting and name is forwarded to the CUBIN generator
  /// to use in its logging mechanisms.
  OwnedCubin convertPtxToCubin(std::string& ptx, Location loc, StringRef name);

  /// Translates llvmModule to cubin and returns the result as attribute.
  StringAttr translateGPUModuleToCubinAnnotation(std::string& ptx, Location loc,
                                                 StringRef name);
  CubinGenerator cubinGenerator;
  int cc_major_;
  int cc_minor_;
};

}  // anonymous namespace

OwnedCubin GpuKernelToCubinPassV2::compilePtxToCubinForTesting(
    const std::string& ptx, Location, StringRef) {
  const char data[] = "CUBIN";
  return std::make_unique<std::vector<char>>(data, data + sizeof(data) - 1);
}

OwnedCubin GpuKernelToCubinPassV2::convertPtxToCubin(std::string& ptx,
                                                     Location loc,
                                                     StringRef name) {
  return cubinGenerator(ptx, loc, name);
}

StringAttr GpuKernelToCubinPassV2::translateGPUModuleToCubinAnnotation(
    std::string& ptx, Location loc, StringRef name) {
  auto cubin = convertPtxToCubin(ptx, loc, name);
  if (!cubin) return {};
  return StringAttr::get({cubin->data(), cubin->size()}, loc->getContext());
}

std::unique_ptr<OpPassBase<gpu::GPUModuleOp>>
createConvertGPUKernelToCubinV2Pass(CubinGenerator cubinGenerator, int cc_major,
                                    int cc_minor) {
  return std::make_unique<GpuKernelToCubinPassV2>(cubinGenerator, cc_major,
                                                  cc_minor);
}

static PassRegistration<GpuKernelToCubinPassV2> pass(
    "test-kernel-to-cubin-v2",
    "Convert all kernel functions to CUDA cubin blobs, reference to xla");

}  // namespace disc_ral
}  // namespace mlir

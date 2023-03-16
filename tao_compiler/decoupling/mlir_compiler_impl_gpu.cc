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

#include "decoupling/mlir_compiler_impl_gpu.h"

#include "cuda.h"

namespace tensorflow {
namespace tao {

#define RETURN_ON_CUDA_ERROR(expr, msg) \
  {                                     \
    auto _cuda_error = (expr);          \
    if (_cuda_error != CUDA_SUCCESS) {  \
      return errors::Internal(msg);     \
    }                                   \
  }

struct CompilerMLIR_GPU::Impl {
  mlir::disc_ral::GpuDeviceInfo device_context;
};

CompilerMLIR_GPU::CompilerMLIR_GPU() : impl_(new Impl) {}

CompilerMLIR_GPU::~CompilerMLIR_GPU() {}

std::string CompilerMLIR_GPU::DefaultDevice() { return "gpu"; }

Status CompilerMLIR_GPU::Init(const TaoCompilerInput& input,
                              const string& output_file) {
  CUdevice device;
  CUcontext context;
  auto& ctx = impl_->device_context;
  RETURN_ON_CUDA_ERROR(cuInit(0), "cuInit");
  RETURN_ON_CUDA_ERROR(cuDeviceGet(&device, ctx.device_ordinal), "cuDeviceGet");
  RETURN_ON_CUDA_ERROR(cuCtxCreate(&context, 0, device), "cuCtxCreate");
  RETURN_ON_CUDA_ERROR(
      cuDeviceComputeCapability(&ctx.cc_major, &ctx.cc_minor, device),
      "cuDeviceComputeCapability");
  RETURN_ON_CUDA_ERROR(
      cuDeviceGetAttribute(&ctx.sm_count,
                           CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device),
      "cuDeviceGetAttribute (MULTIPROCESSOR_COUNT)");
  RETURN_ON_CUDA_ERROR(
      cuDeviceGetAttribute(&ctx.max_threads_per_sm,
                           CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                           device),
      "cuDeviceGetAttribute (MAX_THREADS_PER_MULTIPROCESSOR)");
  // RETURN_ON_CUDA_ERROR(
  //     cuDeviceGetAttribute(&ctx.max_threads_per_block,
  //                          CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
  //                          device),
  //     "cuDeviceGetAttribute (MAX_THREADS_PER_BLOCK)");
  return tsl::OkStatus();
}

Status CompilerMLIR_GPU::FillDeviceInfo(
    mlir::disc_ral::DISCLoweringOptions& options) {
  options.gpu_info = impl_->device_context;
  return tsl::OkStatus();
}

}  // namespace tao
}  // namespace tensorflow

static bool InitModule() {
  tensorflow::tao::CompilerBase::RegisterCompilerFactory(
      "MLIR_GPU", []() -> std::unique_ptr<tensorflow::tao::CompilerBase> {
        return absl::make_unique<tensorflow::tao::CompilerMLIR_GPU>();
      });
  return true;
}
static bool module_initialized = InitModule();

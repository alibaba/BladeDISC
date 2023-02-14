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

#include "decoupling/mlir_compiler_impl_rocm.h"

#define CUDA_SUCCESS hipSuccess
#include "tensorflow/stream_executor/rocm/rocm_driver_wrapper.h"

namespace tensorflow {
namespace tao {

#define RETURN_ON_CUDA_ERROR(expr, msg) \
  {                                     \
    auto _cuda_error = (expr);          \
    if (_cuda_error != CUDA_SUCCESS) {  \
      return errors::Internal(msg);     \
    }                                   \
  }

struct CompilerMLIR_DCU::Impl {
  mlir::disc_ral::GpuDeviceInfo device_context;
};

CompilerMLIR_DCU::CompilerMLIR_DCU() : impl_(new Impl) {}

CompilerMLIR_DCU::~CompilerMLIR_DCU() {}

std::string CompilerMLIR_DCU::DefaultDevice() { return "gpu"; }

Status CompilerMLIR_DCU::Init(const TaoCompilerInput& input,
                              const string& output_file) {
  hipDevice_t device;
  hipCtx_t context;
  auto& ctx = impl_->device_context;
  RETURN_ON_CUDA_ERROR(hipInit(0), "hipInit");
  RETURN_ON_CUDA_ERROR(hipDeviceGet(&device, ctx.device_ordinal),
                       "hipDeviceGet");
  RETURN_ON_CUDA_ERROR(hipCtxCreate(&context, 0, device), "hipCtxCreate");
  RETURN_ON_CUDA_ERROR(
      hipDeviceComputeCapability(&ctx.cc_major, &ctx.cc_minor, device),
      "hipDeviceComputeCapability");
  VLOG(2) << "Finish rocm init";
  return Status::OK();
}

Status CompilerMLIR_DCU::FillDeviceInfo(
    mlir::disc_ral::DISCLoweringOptions& options) {
  options.gpu_info = impl_->device_context;
  return Status::OK();
}

}  // namespace tao
}  // namespace tensorflow

static bool InitModule() {
  tensorflow::tao::CompilerBase::RegisterCompilerFactory(
      "MLIR_GPU", []() -> std::unique_ptr<tensorflow::tao::CompilerBase> {
        return absl::make_unique<tensorflow::tao::CompilerMLIR_DCU>();
      });
  return true;
}
static bool module_initialized = InitModule();

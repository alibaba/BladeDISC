#include "tensorflow/compiler/decoupling/mlir_compiler_impl_gpu.h"

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
  return Status::OK();
}

Status CompilerMLIR_GPU::FillDeviceInfo(
    mlir::disc_ral::DISCLoweringOptions& options) {
  options.gpu_info = impl_->device_context;
  return Status::OK();
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

#include "tao_bridge/kernels/platform_info.h"

namespace tensorflow {
namespace tao {

Status PlatformInfoFromContext(OpKernelConstruction* ctx,
                               PlatformInfo* result) {
  DeviceType device_type = ctx->device_type();
  se::Platform::Id platform_id = nullptr;

  if (ctx->device_type() == DeviceType(DEVICE_CPU)) {
    platform_id = se::host::kHostPlatformId;
  } else if (ctx->device_type() == DeviceType(DEVICE_GPU)) {
    platform_id = ctx->device()
                      ->tensorflow_gpu_device_info()
                      ->stream->parent()
                      ->platform()
                      ->id();
  }

  *result = PlatformInfo(device_type, platform_id);
  return Status::OK();
}

}  // namespace tao
}  // namespace tensorflow
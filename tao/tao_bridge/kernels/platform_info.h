#ifndef TAO_TAO_BRIDGE_KERNELS_PLATFORM_INFO_H_
#define TAO_TAO_BRIDGE_KERNELS_PLATFORM_INFO_H_

#include "tao_bridge/executable.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/stream_executor/host/host_platform_id.h"
#include "tensorflow/stream_executor/platform.h"

namespace tensorflow {
namespace tao {

class PlatformInfo;
Status PlatformInfoFromContext(OpKernelConstruction* ctx, PlatformInfo* result);
class PlatformInfo {
 public:
  PlatformInfo() : device_type_("") {}
  explicit PlatformInfo(const DeviceType device_type,
                        se::Platform::Id platform_id)
      : device_type_(device_type), platform_id_(platform_id) {}

  PlatformInfo& operator=(PlatformInfo&& other) = default;

  DeviceType device_type() const { return device_type_; }

  void set_device_type(const char* type) { device_type_ = DeviceType(type); }

  // This is equal to xla_device_metadata()->platform()->id() if
  // xla_device_metadata() is not nullptr.
  se::Platform::Id platform_id() const { return platform_id_; }

 private:
  DeviceType device_type_;
  se::Platform::Id platform_id_;

  TF_DISALLOW_COPY_AND_ASSIGN(PlatformInfo);
};
/**
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
**/
}  //  namespace tao
}  //  namespace tensorflow

#endif  //  TAO_TAO_BRIDGE_KERNELS_PLATFORM_INFO_H_
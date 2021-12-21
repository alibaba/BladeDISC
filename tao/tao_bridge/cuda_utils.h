#ifndef TAO_CUDA_UTILS_H_
#define TAO_CUDA_UTILS_H_

#include <string>

namespace tensorflow {

class OpKernelConstruction;

namespace tao {
namespace cuda_utils {

// Get UUID of the device where the kernel is placed on.
std::string GetGpuDeviceUUID(const std::string bus_id);

}  // namespace cuda_utils
}  // namespace tao
}  // namespace tensorflow

#endif  // TAO_CUDA_UTILS_H_

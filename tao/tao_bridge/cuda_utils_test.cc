#include "tao_bridge/cuda_utils.h"

#include <dlfcn.h>

#include "gtest/gtest.h"

namespace tensorflow {
namespace tao {
namespace cuda_utils {

namespace {

#ifdef TAO_CPU_ONLY

TEST(CudaUtilsTest, TestGetGpuDeviceUUID_CPU) {
  ASSERT_EQ(GetGpuDeviceUUID("any bus_id"), "");
}

#else

void GetGpuPCIBusID(std::string* bus_id) {
  void* cuda_handle = nullptr;
  // CUDA suffixes in priority order
  const auto cuda_suffixes = {"", ".10.1", ".10.0", ".9.0"};
  for (const std::string& suffix : cuda_suffixes) {
    const auto sopath = "libcudart.so" + suffix;
    cuda_handle = dlopen(sopath.c_str(), RTLD_LAZY);
    if (cuda_handle) {
      break;
    }
  }

  if (!cuda_handle) {
    FAIL() << "Couldn't load the CUDA runtime.";
  }

  typedef int cudaError_t;
  auto cudaDeviceGetPCIBusId =
      reinterpret_cast<cudaError_t (*)(char*, int, int)>(
          dlsym(cuda_handle, "cudaDeviceGetPCIBusId"));

  auto cudaGetLastError = reinterpret_cast<cudaError_t (*)()>(
      dlsym(cuda_handle, "cudaGetLastError"));

  // At most 13 chars according to
  // https://docs.nvidia.com/cuda/archive/9.0/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gea264dad3d8c4898e0b82213c0253def
  char pciBusId[13];
  int err = cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), 0);
  if (err != 0) {
    cudaGetLastError();
    FAIL() << "Couldn't get PIC bus ID for GPU 0.";
  }
  *bus_id = pciBusId;
}

TEST(CudaUtilsTest, TestGetGpuDeviceUUID_GPU) {
  std::string pci_bus_id;
  GetGpuPCIBusID(&pci_bus_id);
  auto device_uuid = GetGpuDeviceUUID(pci_bus_id);
  ASSERT_NE(device_uuid, "");
}

#endif  // TAO_CPU_ONLY

}  // namespace

}  // namespace cuda_utils
}  // namespace tao
}  // namespace tensorflow

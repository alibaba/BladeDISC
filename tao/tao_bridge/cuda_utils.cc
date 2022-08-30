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

#include "tao_bridge/cuda_utils.h"

#include <dlfcn.h>

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace tao {
namespace cuda_utils {

#if defined(TAO_CPU_ONLY) || defined(TENSORFLOW_USE_ROCM)

std::string GetGpuDeviceUUID(const std::string bus_id) { return ""; }

#else

typedef int nvmlReturn_t;
typedef void* nvmlDevice_t;

std::string GetGpuDeviceUUID(const std::string bus_id) {
  void* nvml_handle = nullptr;
  const std::string nvml_suffixes[2] = {"", ".1"};
  for (const std::string& suffix : nvml_suffixes) {
    std::string full_name = "libnvidia-ml.so" + suffix;
    nvml_handle = dlopen(full_name.c_str(), RTLD_LAZY);
    if (nvml_handle != nullptr) {
      break;
    }
  }
  if (nvml_handle == nullptr) {
    LOG(WARNING) << "Could not load libnvidia-ml.so and libnvidia-ml.so.1 .";
    return "";
  }
  // Load NVML functions.
  auto nvmlInit =
      reinterpret_cast<nvmlReturn_t (*)()>(dlsym(nvml_handle, "nvmlInit"));

  auto nvmlDeviceGetHandleByPciBusId =
      reinterpret_cast<nvmlReturn_t (*)(const char*, nvmlDevice_t*)>(
          dlsym(nvml_handle, "nvmlDeviceGetHandleByPciBusId"));

  auto nvmlDeviceGetUUID =
      reinterpret_cast<nvmlReturn_t (*)(nvmlDevice_t, char*, unsigned int)>(
          dlsym(nvml_handle, "nvmlDeviceGetUUID"));

  auto nvmlErrorString = reinterpret_cast<const char* (*)(nvmlReturn_t)>(
      dlsym(nvml_handle, "nvmlErrorString"));

  int err = nvmlInit();
  if (err != 0) {
    LOG(WARNING) << "Error when initializing NVML: " << nvmlErrorString(err);
    return "";
  }

  // Get an NVML device handle
  nvmlDevice_t nvml_device;
  err = nvmlDeviceGetHandleByPciBusId(bus_id.c_str(), &nvml_device);
  if (err != 0) {
    LOG(WARNING) << "NVML error when getting device from pciBusId: "
                 << nvmlErrorString(err);
    return "";
  }

  // Get a UUID from the handle
  // At most 80 chars according to
  // https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries_1g84dca2d06974131ccec1651428596191
  char uuid[80];
  err = nvmlDeviceGetUUID(nvml_device, uuid, sizeof(uuid));
  if (err != 0) {
    LOG(WARNING) << "NVML error when getting uuid from device: "
                 << nvmlErrorString(err);
    return "";
  }
  VLOG(1) << "Device UUID: " << uuid;
  return uuid;
}

#endif  // TAO_CPU_ONLY

}  // namespace cuda_utils
}  // namespace tao
}  // namespace tensorflow

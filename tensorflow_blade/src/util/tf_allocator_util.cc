// Copyright 2022 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tf_allocator_util.h"

#include <algorithm>
#include <cstdlib>

#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_manager.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#include "tensorflow/core/platform/logging.h"

// NB(xiafei.qiuxf): Since tf 2.5, TfGpuId and PlatformGpuId are renamed to
//                   TfDeviceId and PlatformDeviceId respectively.
#if TF_MAJOR == 2 && TF_MINOR > 4
using TfGpuId = tensorflow::TfDeviceId;
using PlatformGpuId = tensorflow::PlatformDeviceId;
constexpr auto TfToPlatformDeviceId_Func =
    &tensorflow::GpuIdManager::TfToPlatformDeviceId;
#else
using tensorflow::PlatformGpuId;
using tensorflow::TfGpuId;
constexpr auto TfToPlatformDeviceId_Func =
    &tensorflow::GpuIdManager::TfToPlatformGpuId;
#endif

using tensorflow::GPUOptions;
using tensorflow::GPUProcessState;

namespace tf_blade {
namespace util {

std::pair<TfGpuId, PlatformGpuId> GetFirstValidDeviceId() {
  for (int tf_gpu_id_value = 0; tf_gpu_id_value < 100; ++tf_gpu_id_value) {
    TfGpuId tf_gpu_id(tf_gpu_id_value);
    PlatformGpuId platform_gpu_id;
    auto s = TfToPlatformDeviceId_Func(tf_gpu_id, &platform_gpu_id);
    if (s.ok()) {
      VLOG(2) << "Found TF GPU " << tf_gpu_id.value() << " at cuda device "
              << platform_gpu_id.value();
      return std::make_pair(tf_gpu_id, platform_gpu_id);
    }
  }
  LOG(ERROR) << "Could not find any TF GPUs";
  return std::make_pair(TfGpuId(-1), PlatformGpuId(-1));
}

std::pair<int, Allocator*> GetDeviceAndAllocator(int device_id) {
  Allocator* dev_allocator = nullptr;
  int cuda_device_id = -1;
  TfGpuId tf_gpu_id;
  PlatformGpuId platform_gpu_id;
  if (device_id < 0) {
    // If device is not set, use the first found GPU device for the conversion.
    std::tie(tf_gpu_id, platform_gpu_id) = GetFirstValidDeviceId();
    cuda_device_id = platform_gpu_id.value();
  } else {
    cuda_device_id = device_id;
    tf_gpu_id = TfGpuId(device_id);
  }
  if (cuda_device_id >= 0) {
    GPUOptions gpu_options;
    // If the TF to Cuda gpu id mapping exist, the device and corresponding
    // allocator must have been initialized already, so the
    // GetGPUAllocator() call won't create a new allocator.
    dev_allocator =
        GPUProcessState::singleton()->GetGPUAllocator(gpu_options, tf_gpu_id, 1
#if TF_MAJOR == 2 && TF_MINOR > 4
                                                      ,
                                                      /*peer_gpu_ids*/ {}
#endif
        );
  }
  return std::make_pair(cuda_device_id, dev_allocator);
}

// std::align is not supported, so this method mimic its behavior.
void* Align(uint64_t alignment, uint64_t size, void*& ptr, uint64_t& space) {
  QCHECK_GT(alignment, 0ul) << "alignment must be greater than 0.";
  QCHECK_EQ(0, alignment & (alignment - 1)) << "Alignment must be power of 2.";
  QCHECK_GT(size, 0ul) << "size must be greater than 0.";
  QCHECK(ptr) << "ptr must not be nullptr.";
  QCHECK_GT(space, 0ul) << "space must be greater than 0.";
  const uintptr_t ptr_val = reinterpret_cast<uintptr_t>(ptr);
  QCHECK_GE(ptr_val + space, ptr_val) << "Provided space overflows.";

  if (size > space) return nullptr;
  const uintptr_t aligned_ptr_val = ((ptr_val + alignment - 1) & -alignment);
  if (aligned_ptr_val > ptr_val + space - size) return nullptr;
  ptr = reinterpret_cast<void*>(aligned_ptr_val);
  const uintptr_t diff = aligned_ptr_val - ptr_val;
  space -= diff;
  return ptr;
}

}  // namespace util
}  // namespace tf_blade

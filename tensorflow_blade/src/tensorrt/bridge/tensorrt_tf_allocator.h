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

#ifndef __TENSORRT_ALLOCATOR_H__
#define __TENSORRT_ALLOCATOR_H__

#include <unordered_map>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/mutex.h"
#if TF_MAJOR == 2
#include "tensorflow/core/platform/status.h"
#else
#include "tensorflow/core/lib/core/status.h"
#endif

#include "NvInfer.h"

// tf-trt's allocator impl -- it's better to maintain our own version

namespace tf_blade {
namespace trt {

using tensorflow::Allocator;
using tensorflow::mutex;
using tensorflow::Status;

class TRTBaseAllocator : public nvinfer1::IGpuAllocator {
  // Base allocator class so we can have a virtual destructor;
 public:
  // python wrapper seems to be not happy with an pure virtual destructor;
  virtual ~TRTBaseAllocator() = default;
};

class TRTDeviceAllocator : public TRTBaseAllocator {
  // Allocator implementation wrapping TF device allocators.
 public:
  TRTDeviceAllocator(Allocator* allocator);

  // TODO(aaroey): base class doesn't have a virtual destructor, work with
  // Nvidia to fix it.
  virtual ~TRTDeviceAllocator() {
    VLOG(2) << "Destroying allocator attached to " << allocator_->Name();
  }
  void* allocate(uint64_t size, uint64_t alignment,
                 uint32_t flags) noexcept override;
  void free(void* memory) noexcept override;

 private:
  mutex mu_;
  Allocator* allocator_;

  // supporting alignment from allocation request requires a map to free
#if TF_MAJOR == 2
  std::unordered_map<void*, void*> mem_map_ TF_GUARDED_BY(mu_);
#else
  std::unordered_map<void*, void*> mem_map_ GUARDED_BY(mu_);
#endif
};

// Allocates device memory for an execution context to execute a TensorRT
// engine and records the relevant information for deallocating the memory when
// the engine finishes execution.
class ContextDeviceMemory {
 public:
  ContextDeviceMemory()
      : execution_context_(nullptr),
        device_memory_allocator_(nullptr),
        device_memory_(nullptr) {}

  ~ContextDeviceMemory() {
    if (device_memory_) {
      device_memory_allocator_->free(device_memory_);
    }
  }

  Status AllocateDeviceMemory(nvinfer1::IExecutionContext* execution_context,
                              TRTBaseAllocator* device_memory_allocator);

 private:
  nvinfer1::IExecutionContext* execution_context_;
  TRTBaseAllocator* device_memory_allocator_;
  void* device_memory_;
};

}  // namespace trt
}  // namespace tf_blade
#endif  //__TENSORRT_ALLOCATOR_H__

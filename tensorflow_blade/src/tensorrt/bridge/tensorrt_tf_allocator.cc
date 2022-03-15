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

#include "tensorrt_tf_allocator.h"

#include <algorithm>
#include <cstdlib>

#include "src/util/tf_allocator_util.h"

#if TF_MAJOR == 2
#include "tensorflow/core/platform/errors.h"
#else
#include "tensorflow/core/lib/core/errors.h"
#endif
#include "tensorflow/core/platform/logging.h"

using tensorflow::AllocationAttributes;
using tensorflow::mutex_lock;

namespace tf_blade {
namespace trt {

void* TRTDeviceAllocator::allocate(uint64_t size, uint64_t alignment,
                                   uint32_t flags) noexcept {
  if (size == 0) return nullptr;
  // WAR for allocator alignment requirement. Certain cuda API calls require GPU
  // memory with alignment to cudaDeviceProp::textureAlignment.
  // See issue #20856
  alignment = 512;
  assert((alignment & (alignment - 1)) == 0);  // zero or a power of 2.
  uint64_t total_size = size + alignment;
  // TODO(aaroey): AllocateRaw takes size_t size as input, so it'll produce
  // unexpected result when TRT tries to allocate more bytes than size_t can
  // carry. Fix this.
  //
  // Fail immediately if allocation fails, rather than waiting 10 seconds and
  // failing then anyway.
  // TensorRT 7 can also switch to a different algorithm for a layer if an
  // algorithm uses too much memory. If we don't fail immediately building the
  // engine can be *very* slow with TensorRT7 when GPU memory is limited.
  AllocationAttributes attributes;
#if TF_MAJOR == 2
  attributes.retry_on_failure = false;
#endif
  void* mem = allocator_->AllocateRaw(alignment, total_size, attributes);
  if (!mem) return nullptr;

  void* alloc_mem = mem;
  QCHECK(tf_blade::util::Align(alignment, size, mem, total_size));
  mutex_lock lock(mu_);
  if (mem != alloc_mem) {
    QCHECK(mem_map_.insert({mem, alloc_mem}).second);
  }
  VLOG(2) << "Allocated " << total_size << " bytes memory @" << alloc_mem
          << "; aligned to " << size << " bytes @" << mem << " with alignment "
          << alignment;
  return mem;
}

TRTDeviceAllocator::TRTDeviceAllocator(Allocator* allocator)
    : allocator_(allocator) {
  VLOG(2) << "Using " << allocator->Name() << " allocator from TensorFlow";
}

void TRTDeviceAllocator::free(void* memory) noexcept {
  mutex_lock lock(mu_);
  VLOG(2) << "Deallocating @ " << memory;
  // allocated memory adjusted for alignment, restore the original pointer
  if (memory) {
    auto alloc_mem = mem_map_.find(memory);
    if (alloc_mem != mem_map_.end()) {
      memory = alloc_mem->second;
      mem_map_.erase(alloc_mem->first);
    }
    allocator_->DeallocateRaw(memory);
  }
}

Status ContextDeviceMemory::AllocateDeviceMemory(
    nvinfer1::IExecutionContext* execution_context,
    TRTBaseAllocator* device_memory_allocator) {
  if (!execution_context || !device_memory_allocator) {
    return tensorflow::errors::Internal(
        "Failed to get valid ExecutionContext or allocator for memory "
        "allocation!");
  }
  execution_context_ = execution_context;
  device_memory_allocator_ = device_memory_allocator;
  device_memory_ = nullptr;
  const size_t device_memory_size =
      execution_context_->getEngine().getDeviceMemorySize();
  VLOG(2) << "Device memory size for TensorRT engine " << device_memory_size;
  if (device_memory_size > 0) {
    device_memory_ = device_memory_allocator_->allocate(device_memory_size,
                                                        /*unused alignment=*/0,
                                                        /*flags=*/0);
    if (device_memory_ == nullptr) {
      return tensorflow::errors::InvalidArgument(
          "Out of GPU memory for execution context");
    }
  }
  execution_context_->setDeviceMemory(device_memory_);

  return Status::OK();
}

}  // namespace trt
}  // namespace tf_blade

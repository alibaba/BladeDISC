//===- cuda_context_impl.h ----------------------===//
//
// Copyright 2020 The PAI Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================

#ifndef RAL_CONTEXT_BASE_CUDA_CUDA_CONTEXT_IMPL_H_
#define RAL_CONTEXT_BASE_CUDA_CUDA_CONTEXT_IMPL_H_

#if GOOGLE_CUDA
#include "cuda.h"
#endif

#if TENSORFLOW_USE_ROCM
#define __HIP_DISABLE_CPP_FUNCTIONS__
#include "rocm/include/hip/hip_runtime.h"
#endif

#include "mlir/ral/context/base/cpu/cpu_context_impl.h"
#include "mlir/ral/device/cpu/cpu_driver.h"
#include "mlir/ral/device/gpu/gpu_driver.h"
#include "mlir/ral/ral_context.h"
#include "third_party/nccl/nccl.h"

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

// Raw cuda ral implementation.

namespace tao {
namespace ral {
namespace gpu {

#if TENSORFLOW_USE_ROCM
using GpuStreamHandle = hipStream_t;
using GpuEventHandle = hipEvent_t;
#else
using GpuStreamHandle = CUstream;
using GpuEventHandle = cudaEvent_t;
#endif

struct BaseCudaContextOption {
  ncclComm_t nccl_comm = nullptr;
  GpuStreamHandle stream = nullptr;
  GpuStreamHandle comm_stream = nullptr;
  int device_ordinal = 0;
  bool use_stream_executor = true;
  bool cache_workspace_mem_across_execution = false;
  std::shared_ptr<Allocator> gpu_allocator;
};

struct EvictionManager {
  using BufferInt64Ptr = int64_t;
  EvictionManager() {
    // Need to read from env
    execution_memory_usage_limit_ = 10ll * 1024ll * 1024ll * 1024ll;
  }

  bool Evict(BufferInt64Ptr memref_buffer) {
    if(reinterpret_cast<void*>(memref_buffer) == nullptr) return true;
    if(current_execution_memory_usage_ < execution_memory_usage_limit_) return false;
    return true;
  }

  std::vector<bool> Evict(std::vector<BufferInt64Ptr>& memref_buffers, std::vector<double>& distance_to_next_usage) {
    std::vector<bool> evicted_memrefs(memref_buffers.size(), false);
    std::vector<std::pair<BufferInt64Ptr, double>> memref_scores;

    if(current_execution_memory_usage_ < execution_memory_usage_limit_)  return evicted_memrefs;
    
    for (int idx=0; idx < memref_buffers.size(); idx++) {
      // Already evicted
      if (reinterpret_cast<void*>(memref_buffers[idx]) == nullptr) {
        evicted_memrefs[idx] = true;
      } else {
        memref_scores.push_back(std::make_pair(idx, memref_size_map_[memref_buffers[idx]] *  distance_to_next_usage[idx]));
      }
    }

    std::sort(memref_scores.begin(), memref_scores.end(),
              [](const std::pair<BufferInt64Ptr, double>& a,
                 const std::pair<BufferInt64Ptr, double>& b) {
                return a.second > b.second;
              });
    
    size_t current_memory_usage = current_execution_memory_usage_;
    for (auto pair : memref_scores) {
      evicted_memrefs[pair.first] = true;
      current_memory_usage -= memref_size_map_[memref_buffers[pair.first]];
      if(current_memory_usage < execution_memory_usage_limit_) break;
    }
    return evicted_memrefs;
  }

  // Track runtime allocation & deallocation actions
  void TrackAlloc(void* buffer, size_t size) {
    auto buffer_int64_ptr = reinterpret_cast<BufferInt64Ptr>(buffer);
    memref_size_map_[buffer_int64_ptr] = size;
    current_execution_memory_usage_ += size;
  }

  void TrackDealloc(void* buffer) {
    auto buffer_int64_ptr = reinterpret_cast<BufferInt64Ptr>(buffer);
    current_execution_memory_usage_ -= memref_size_map_[buffer_int64_ptr];
    memref_size_map_.erase(buffer_int64_ptr);
  }

  private:
    std::unordered_map<BufferInt64Ptr, size_t> memref_size_map_;
    size_t current_execution_memory_usage_ = 0;
    size_t execution_memory_usage_limit_ = 0;

};

std::unique_ptr<BaseContext> MakeBaseCudaContext(
    BaseContextOption& opt, ::tao::ral::cpu::BaseCpuContextOption& cpu_opt,
    ::tao::ral::gpu::BaseCudaContextOption& gpu_opt);

struct BaseCudaExecutionContext
    : public tao::ral::cpu::BaseCpuExecutionContext {
  BaseCudaExecutionContext(BaseContext* ctx);
  ~BaseCudaExecutionContext();

  ncclComm_t getNcclComm();

  GpuStreamHandle getCommStream();

  // We need to sync on the gpu stream before we fetch the first output.
  bool synced = false;
  // all buffer allocated by the gpu_allocator
  std::unordered_map<const_buffer_t, int> device_ptr_map;

  // map int64 -> cudaEvent_t
  std::map<int64_t, GpuEventHandle> async_pair_tokens;

  EvictionManager eviction_manager;

 protected:
  virtual void setOutputDeleter(OutputBufferWrapper& output) override;
};

}  // namespace gpu
}  // namespace ral
}  // namespace tao

#endif  // RAL_CONTEXT_BASE_CUDA_CUDA_CONTEXT_IMPL_H_

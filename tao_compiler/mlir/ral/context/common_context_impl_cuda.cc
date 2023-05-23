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

#include <fcntl.h>

#include <cstring>
#include <numeric>

#include "absl/strings/str_split.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"
#include "mlir/ral/context/common_context_impl.h"
#include "mlir/ral/context/context_util.h"
#include "mlir/ral/context/stream_executor_based_impl.h"
#include "mlir/ral/device/cpu/cpu_driver.h"
#include "mlir/ral/device/gpu/gpu_driver.h"
#include "mlir/ral/ral_base.h"
#include "mlir/ral/ral_helper.h"

namespace tao {
namespace ral {

using tao::ral::gpu::GPUDriver;
const char* kRalGpuD2DCopy = "d2d";
const char* kRalGpuH2DCopy = "h2d";
const char* kRalGpuD2HCopy = "d2h";
const char* kRalGpuPrint = "ral_gpu_print";

void RalGlobalConstantState::onContextFinish(Context* ctx) /* override */ {
  if (process_level_store) {
    bool owned = ConstStoreRegistrar::Instance().unregisterConstStore(
        process_level_store);
    if (!owned) return;
    auto cpu_driver =
        static_cast<cpu::CPUDriver*>(ctx->getDriver(cpu::CPUDriver::name()));
    auto gpu_driver =
        static_cast<gpu::GPUDriver*>(ctx->getDriver(gpu::GPUDriver::name()));
    for (auto& e : process_level_store->state.host_constants) {
      cpu_driver->raw_dealloc(ctx, e.second.first);
    }
    for (auto& e : process_level_store->state.device_constants) {
      gpu_driver->raw_dealloc(ctx, e.second.first);
    }
    delete process_level_store;
    return;
  }
  // Skip if not process level const store since the context will free these
  // const buffer correctly.
}

static inline buffer_t ral_base_cuda_const_cuda_internal(
    ExecutionContext* ctx, void* stream_handle, const char* unique_name,
    int32_t unique_index_in_module, buffer_shape_t*& shape) {
  auto* state =
      ctx->getResource<RalGlobalConstantState>(kRalGlobalConstantState);
  // if process-level const store is enabled, use it instead of the context
  // level store.
  bool use_process_store = false;
  if (state->process_level_store) {
    state = &(state->process_level_store->state);
    use_process_store = true;
  }

  // fast path: using a unique const index to do look up.
  // Note that the const index is assigned to each const at compile time.
  // The index is unique within the compiled module level.
  if (auto item = state->getDeviceConstByIndex(unique_index_in_module)) {
    shape = &item->second;
    return item->first;
  }

  {
    std::lock_guard<std::mutex> lock(state->mu);
    std::string key(unique_name);
    TAO_VLOG(2) << "unique_name: " << key;
    auto it = state->device_constants.find(key);
    if (it == state->device_constants.end()) {
      int64_t width_in_bytes = 0;
      buffer_shape_t dim_sizes =
          GetShapeFromConstUniqueName(ctx, unique_name, &width_in_bytes);
      // alloc, get value from metadata file, and then memcpy
      const std::string* hex_str_ptr;
      if (!state->metadata->getDeviceConstant(key, hex_str_ptr)) {
        std::string msg =
            "const unique_name " + key + "not found in metadata file";
        ctx->signalError(Context::FAILURE, msg);
      }
      auto data = fromHex(*hex_str_ptr);
      auto bytes = data.size();
      int64_t num_elements = std::accumulate(dim_sizes.begin(), dim_sizes.end(),
                                             1, std::multiplies<int64_t>());
      if (bytes < num_elements * width_in_bytes) {
        // isSplat
        bytes = num_elements * width_in_bytes;
        auto splat_data = data;
        for (int64_t i = 0; i < num_elements - 1; ++i) {
          std::copy(splat_data.begin(), splat_data.end(),
                    std::back_inserter(data));
        }
      }
      TAO_VLOG(2) << "data.size: " << bytes;
      auto gpu_driver = ctx->getDriver<GPUDriver>(GPUDriver::name());

      buffer_t device_ptr =
          use_process_store ? gpu_driver->raw_alloc(ctx->getContext(), bytes)
                            : gpu_driver->alloc_persistent(ctx, bytes);
      state->metadata->releaseDeviceConstant(key);

      gpu_driver->h2d(ctx, stream_handle, data.data(), device_ptr, bytes);
      // Insert a sync to make sure copy is done before host buffer is freed.
      gpu_driver->syncOnStream(ctx, stream_handle);
      it = state->device_constants
               .insert(
                   std::make_pair(key, std::make_pair(device_ptr, dim_sizes)))
               .first;
      state->setDeviceConstByIndex(unique_index_in_module,
                                   std::make_pair(device_ptr, dim_sizes));
    }
    shape = &it->second.second;
    return it->second.first;
  }
}

template <typename T, int N>
MemRefType<T, N> ral_base_cuda_const_cuda(ExecutionContext* ctx,
                                          void* stream_handle,
                                          const char* unique_name,
                                          int32_t unique_index_in_module) {
  buffer_shape_t* shape = nullptr;
  buffer_t ptr = ral_base_cuda_const_cuda_internal(
      ctx, stream_handle, unique_name, unique_index_in_module, shape);
  if (!shape || shape->size() != N) {
    ctx->signalError(Context::FAILURE,
                     "unexpected shape string in unique_name");
  }

  return assignMemRef<T, N>(ptr, *shape);
}

template <typename T>
MemRefType<T, 0> ral_base_cuda_const_cuda_0d(ExecutionContext* ctx,
                                             void* stream_handle,
                                             const char* unique_name,
                                             int32_t unique_index_in_module) {
  buffer_shape_t* shape = nullptr;
  buffer_t ptr = ral_base_cuda_const_cuda_internal(
      ctx, stream_handle, unique_name, unique_index_in_module, shape);
  if (!shape || shape->size() != 0) {
    ctx->signalError(Context::FAILURE,
                     "unexpected shape string in unique_name");
  }
  return assignMemRef_0d<T>(ptr);
}

#define RAL_REGISTER_CONST_CUDA_FUNC(T, N)                                  \
  template MemRefType<T, N> ral_base_cuda_const_cuda<T, N>(                 \
      ExecutionContext * ctx, void* stream_handle, const char* unique_name, \
      int32_t unique_index_in_module);                                      \
  TAO_RAL_API(tao::ral::kRalCudaConst, "gpu", ral_base_cuda_const_cuda<T, N>);

#define RAL_REGISTER_CONST_CUDA_FUNC_0D(T)                                  \
  template MemRefType<T, 0> ral_base_cuda_const_cuda_0d<T>(                 \
      ExecutionContext * ctx, void* stream_handle, const char* unique_name, \
      int32_t unique_index_in_module);                                      \
  TAO_RAL_API(tao::ral::kRalCudaConst, "gpu", ral_base_cuda_const_cuda_0d<T>);

RAL_REGISTER_CONST_CUDA_FUNC_0D(double);
RAL_REGISTER_CONST_CUDA_FUNC_0D(float);
RAL_REGISTER_CONST_CUDA_FUNC_0D(int8_t);
RAL_REGISTER_CONST_CUDA_FUNC_0D(uint8_t);
RAL_REGISTER_CONST_CUDA_FUNC_0D(int16_t);
RAL_REGISTER_CONST_CUDA_FUNC_0D(uint16_t);
RAL_REGISTER_CONST_CUDA_FUNC_0D(int32_t);
RAL_REGISTER_CONST_CUDA_FUNC_0D(uint32_t);
RAL_REGISTER_CONST_CUDA_FUNC_0D(int64_t);
RAL_REGISTER_CONST_CUDA_FUNC_0D(bool);
RAL_REGISTER_CONST_CUDA_FUNC(double, 1);
RAL_REGISTER_CONST_CUDA_FUNC(double, 2);
RAL_REGISTER_CONST_CUDA_FUNC(double, 3);
RAL_REGISTER_CONST_CUDA_FUNC(double, 4);
RAL_REGISTER_CONST_CUDA_FUNC(double, 5);
RAL_REGISTER_CONST_CUDA_FUNC(double, 6);
RAL_REGISTER_CONST_CUDA_FUNC(double, 7);
RAL_REGISTER_CONST_CUDA_FUNC(double, 8);
RAL_REGISTER_CONST_CUDA_FUNC(float, 1);
RAL_REGISTER_CONST_CUDA_FUNC(float, 2);
RAL_REGISTER_CONST_CUDA_FUNC(float, 3);
RAL_REGISTER_CONST_CUDA_FUNC(float, 4);
RAL_REGISTER_CONST_CUDA_FUNC(float, 5);
RAL_REGISTER_CONST_CUDA_FUNC(float, 6);
RAL_REGISTER_CONST_CUDA_FUNC(float, 7);
RAL_REGISTER_CONST_CUDA_FUNC(float, 8);
RAL_REGISTER_CONST_CUDA_FUNC(int8_t, 1);
RAL_REGISTER_CONST_CUDA_FUNC(int8_t, 2);
RAL_REGISTER_CONST_CUDA_FUNC(int8_t, 3);
RAL_REGISTER_CONST_CUDA_FUNC(int8_t, 4);
RAL_REGISTER_CONST_CUDA_FUNC(int8_t, 5);
RAL_REGISTER_CONST_CUDA_FUNC(int8_t, 6);
RAL_REGISTER_CONST_CUDA_FUNC(int8_t, 7);
RAL_REGISTER_CONST_CUDA_FUNC(int8_t, 8);
RAL_REGISTER_CONST_CUDA_FUNC(uint8_t, 1);
RAL_REGISTER_CONST_CUDA_FUNC(uint8_t, 2);
RAL_REGISTER_CONST_CUDA_FUNC(uint8_t, 3);
RAL_REGISTER_CONST_CUDA_FUNC(uint8_t, 4);
RAL_REGISTER_CONST_CUDA_FUNC(uint8_t, 5);
RAL_REGISTER_CONST_CUDA_FUNC(uint8_t, 6);
RAL_REGISTER_CONST_CUDA_FUNC(uint8_t, 7);
RAL_REGISTER_CONST_CUDA_FUNC(uint8_t, 8);
RAL_REGISTER_CONST_CUDA_FUNC(int16_t, 1);
RAL_REGISTER_CONST_CUDA_FUNC(int16_t, 2);
RAL_REGISTER_CONST_CUDA_FUNC(int16_t, 3);
RAL_REGISTER_CONST_CUDA_FUNC(int16_t, 4);
RAL_REGISTER_CONST_CUDA_FUNC(int16_t, 5);
RAL_REGISTER_CONST_CUDA_FUNC(int16_t, 6);
RAL_REGISTER_CONST_CUDA_FUNC(int16_t, 7);
RAL_REGISTER_CONST_CUDA_FUNC(int16_t, 8);
RAL_REGISTER_CONST_CUDA_FUNC(uint16_t, 1);
RAL_REGISTER_CONST_CUDA_FUNC(uint16_t, 2);
RAL_REGISTER_CONST_CUDA_FUNC(uint16_t, 3);
RAL_REGISTER_CONST_CUDA_FUNC(uint16_t, 4);
RAL_REGISTER_CONST_CUDA_FUNC(uint16_t, 5);
RAL_REGISTER_CONST_CUDA_FUNC(uint16_t, 6);
RAL_REGISTER_CONST_CUDA_FUNC(uint16_t, 7);
RAL_REGISTER_CONST_CUDA_FUNC(uint16_t, 8);
RAL_REGISTER_CONST_CUDA_FUNC(int32_t, 1);
RAL_REGISTER_CONST_CUDA_FUNC(int32_t, 2);
RAL_REGISTER_CONST_CUDA_FUNC(int32_t, 3);
RAL_REGISTER_CONST_CUDA_FUNC(int32_t, 4);
RAL_REGISTER_CONST_CUDA_FUNC(int32_t, 5);
RAL_REGISTER_CONST_CUDA_FUNC(int32_t, 6);
RAL_REGISTER_CONST_CUDA_FUNC(int32_t, 7);
RAL_REGISTER_CONST_CUDA_FUNC(int32_t, 8);
RAL_REGISTER_CONST_CUDA_FUNC(uint32_t, 1);
RAL_REGISTER_CONST_CUDA_FUNC(uint32_t, 2);
RAL_REGISTER_CONST_CUDA_FUNC(uint32_t, 3);
RAL_REGISTER_CONST_CUDA_FUNC(uint32_t, 4);
RAL_REGISTER_CONST_CUDA_FUNC(uint32_t, 5);
RAL_REGISTER_CONST_CUDA_FUNC(uint32_t, 6);
RAL_REGISTER_CONST_CUDA_FUNC(uint32_t, 7);
RAL_REGISTER_CONST_CUDA_FUNC(uint32_t, 8);
RAL_REGISTER_CONST_CUDA_FUNC(int64_t, 1);
RAL_REGISTER_CONST_CUDA_FUNC(int64_t, 2);
RAL_REGISTER_CONST_CUDA_FUNC(int64_t, 3);
RAL_REGISTER_CONST_CUDA_FUNC(int64_t, 4);
RAL_REGISTER_CONST_CUDA_FUNC(int64_t, 5);
RAL_REGISTER_CONST_CUDA_FUNC(int64_t, 6);
RAL_REGISTER_CONST_CUDA_FUNC(int64_t, 7);
RAL_REGISTER_CONST_CUDA_FUNC(int64_t, 8);

RAL_REGISTER_CONST_CUDA_FUNC(bool, 1);
RAL_REGISTER_CONST_CUDA_FUNC(bool, 2);
RAL_REGISTER_CONST_CUDA_FUNC(bool, 3);
RAL_REGISTER_CONST_CUDA_FUNC(bool, 4);
RAL_REGISTER_CONST_CUDA_FUNC(bool, 5);
RAL_REGISTER_CONST_CUDA_FUNC(bool, 6);
RAL_REGISTER_CONST_CUDA_FUNC(bool, 7);
RAL_REGISTER_CONST_CUDA_FUNC(bool, 8);

#ifdef TAO_RAL_USE_STREAM_EXECUTOR
RAL_REGISTER_CONST_CUDA_FUNC_0D(Eigen::half);
RAL_REGISTER_CONST_CUDA_FUNC(Eigen::half, 1);
RAL_REGISTER_CONST_CUDA_FUNC(Eigen::half, 2);
RAL_REGISTER_CONST_CUDA_FUNC(Eigen::half, 3);
RAL_REGISTER_CONST_CUDA_FUNC(Eigen::half, 4);
RAL_REGISTER_CONST_CUDA_FUNC(Eigen::half, 5);
RAL_REGISTER_CONST_CUDA_FUNC(Eigen::half, 6);
RAL_REGISTER_CONST_CUDA_FUNC(Eigen::half, 7);
RAL_REGISTER_CONST_CUDA_FUNC(Eigen::half, 8);
#endif  // TAO_RAL_USE_STREAM_EXECUTOR

static inline void ral_base_cuda_d2d_copy_memref_impl(ExecutionContext* ctx,
                                                      void* stream_handle,
                                                      void* from, void* to,
                                                      int64_t bytes) {
  auto gpu_driver = ctx->getDriver<GPUDriver>(GPUDriver::name());
  gpu_driver->d2d(ctx, stream_handle, from, to, bytes);
}

template <typename T, int N>
void ral_base_cuda_d2d_copy_memref(ExecutionContext* ctx, void* stream_handle,
                                   MemRefType<T, N> from, MemRefType<T, N> to) {
  int64_t bytes = sizeof(T);
  for (int i = 0; i < N; ++i) bytes *= from.sizes[i];
  ral_base_cuda_d2d_copy_memref_impl(ctx, stream_handle, from.data, to.data,
                                     bytes);
}

template <typename T>
void ral_base_cuda_d2d_copy_memref_0d(ExecutionContext* ctx,
                                      void* stream_handle,
                                      MemRefType<T, 0> from,
                                      MemRefType<T, 0> to) {
  int64_t bytes = sizeof(T);
  ral_base_cuda_d2d_copy_memref_impl(ctx, stream_handle, from.data, to.data,
                                     bytes);
}

template <typename T>
static inline void ral_base_cuda_print_memref_impl(ExecutionContext* ctx,
                                                   void* stream_handle,
                                                   void* from, void* to,
                                                   int64_t bytes) {
  for (int64_t i = 0; i < bytes / sizeof(T); ++i) {
    TAO_VLOG(0) << reinterpret_cast<T*>(from)[i] << ", ";
  }
  TAO_VLOG(0) << "] " << std::endl;
  memcpy(to, from, bytes);
}

template <typename T, int N>
void ral_base_cuda_print_memref(ExecutionContext* ctx, void* stream_handle,
                                MemRefType<T, N> from, MemRefType<T, N> to) {
  int64_t bytes = sizeof(T);
  TAO_VLOG(0) << "DebugPrint F32 shape:";
  for (int i = 0; i < N; ++i) {
    auto size = from.sizes[i];
    bytes *= size;
    TAO_VLOG(0) << size << ",";
  }
  TAO_VLOG(0) << " [" << std::endl;
  ral_base_cuda_print_memref_impl<T>(ctx, stream_handle, from.data, to.data,
                                     bytes);
}

template <typename T>
void ral_base_cuda_print_memref_0d(ExecutionContext* ctx, void* stream_handle,
                                   MemRefType<T, 0> from, MemRefType<T, 0> to) {
  int64_t bytes = sizeof(T);
  TAO_VLOG(0) << "DebugPrint F32 shape: ";
  TAO_VLOG(0) << " [" << std::endl;
  ral_base_cuda_print_memref_impl<T>(ctx, stream_handle, from.data, to.data,
                                     bytes);
}

template <typename T>
void ral_base_cuda_print_i32_memref_0d(ExecutionContext* ctx,
                                       void* stream_handle,
                                       MemRefType<T, 0> from,
                                       MemRefType<T, 0> to) {
  int64_t bytes = sizeof(T);
  TAO_VLOG(0) << "DebugPrint I32 shape:scalar: [" << std::endl;
  ral_base_cuda_print_i32_memref_impl(ctx, stream_handle, from.data, to.data,
                                      bytes);
}

static inline void ral_base_cuda_h2d_copy_memref_impl(ExecutionContext* ctx,
                                                      void* stream_handle,
                                                      void* from, void* to,
                                                      int64_t bytes) {
  TAO_VLOG(1) << "from: " << from << " to: " << to;
  auto gpu_driver = ctx->getDriver<GPUDriver>(GPUDriver::name());
  gpu_driver->h2d(ctx, stream_handle, from, to, bytes);
}

template <typename T, int N>
void ral_base_cuda_h2d_copy_memref(ExecutionContext* ctx, void* stream_handle,
                                   MemRefType<T, N> from, MemRefType<T, N> to) {
  int64_t bytes = sizeof(T);
  for (int i = 0; i < N; ++i) bytes *= from.sizes[i];
  ral_base_cuda_h2d_copy_memref_impl(ctx, stream_handle, from.data, to.data,
                                     bytes);
}

template <typename T>
void ral_base_cuda_h2d_copy_memref_0d(ExecutionContext* ctx,
                                      void* stream_handle,
                                      MemRefType<T, 0> from,
                                      MemRefType<T, 0> to) {
  int64_t bytes = sizeof(T);
  TAO_VLOG(1) << "ral_base_cuda_h2d_copy_memref_0d: " << bytes;
  ral_base_cuda_h2d_copy_memref_impl(ctx, stream_handle, from.data, to.data,
                                     bytes);
  TAO_VLOG(1) << "ral_base_cuda_h2d_copy_memref_0d done";
}

static inline void ral_base_cuda_d2h_copy_memref_impl(ExecutionContext* ctx,
                                                      void* stream_handle,
                                                      void* from, void* to,
                                                      int64_t bytes) {
  auto gpu_driver = ctx->getDriver<GPUDriver>(GPUDriver::name());
  gpu_driver->d2h(ctx, stream_handle, from, to, bytes);
}

template <typename T, int N>
void ral_base_cuda_d2h_copy_memref(ExecutionContext* ctx, void* stream_handle,
                                   MemRefType<T, N> from, MemRefType<T, N> to) {
  int64_t bytes = sizeof(T);
  for (int i = 0; i < N; ++i) bytes *= from.sizes[i];
  ral_base_cuda_d2h_copy_memref_impl(ctx, stream_handle, from.data, to.data,
                                     bytes);
}

template <typename T>
void ral_base_cuda_d2h_copy_memref_0d(ExecutionContext* ctx,
                                      void* stream_handle,
                                      MemRefType<T, 0> from,
                                      MemRefType<T, 0> to) {
  int64_t bytes = sizeof(T);
  ral_base_cuda_d2h_copy_memref_impl(ctx, stream_handle, from.data, to.data,
                                     bytes);
}

#define RAL_REGISTER_GPU_COPY_MEMREF_FUNC(T, N)                      \
  template void ral_base_cuda_h2d_copy_memref<T, N>(                 \
      ExecutionContext * ctx, void* stream_handle, MemRefType<T, N>, \
      MemRefType<T, N>);                                             \
  template void ral_base_cuda_d2h_copy_memref<T, N>(                 \
      ExecutionContext * ctx, void* stream_handle, MemRefType<T, N>, \
      MemRefType<T, N>);                                             \
  template void ral_base_cuda_d2d_copy_memref<T, N>(                 \
      ExecutionContext * ctx, void* stream_handle, MemRefType<T, N>, \
      MemRefType<T, N>);                                             \
  TAO_RAL_API(tao::ral::kRalGpuH2DCopy, "gpu",                       \
              ral_base_cuda_h2d_copy_memref<T, N>);                  \
  TAO_RAL_API(tao::ral::kRalGpuD2HCopy, "gpu",                       \
              ral_base_cuda_d2h_copy_memref<T, N>);                  \
  TAO_RAL_API(tao::ral::kRalGpuD2DCopy, "gpu",                       \
              ral_base_cuda_d2d_copy_memref<T, N>);                  \
  TAO_RAL_API(tao::ral::kRalGpuPrint, "gpu", ral_base_cuda_print_memref<T, N>);

#define RAL_REGISTER_GPU_COPY_MEMREF_FUNC_0D(T)                      \
  template void ral_base_cuda_h2d_copy_memref_0d<T>(                 \
      ExecutionContext * ctx, void* stream_handle, MemRefType<T, 0>, \
      MemRefType<T, 0>);                                             \
  template void ral_base_cuda_d2h_copy_memref_0d<T>(                 \
      ExecutionContext * ctx, void* stream_handle, MemRefType<T, 0>, \
      MemRefType<T, 0>);                                             \
  template void ral_base_cuda_d2d_copy_memref_0d<T>(                 \
      ExecutionContext * ctx, void* stream_handle, MemRefType<T, 0>, \
      MemRefType<T, 0>);                                             \
  TAO_RAL_API(tao::ral::kRalGpuH2DCopy, "gpu",                       \
              ral_base_cuda_h2d_copy_memref_0d<T>);                  \
  TAO_RAL_API(tao::ral::kRalGpuD2HCopy, "gpu",                       \
              ral_base_cuda_d2h_copy_memref_0d<T>);                  \
  TAO_RAL_API(tao::ral::kRalGpuD2DCopy, "gpu",                       \
              ral_base_cuda_d2d_copy_memref_0d<T>);                  \
  TAO_RAL_API(tao::ral::kRalGpuPrint, "gpu", ral_base_cuda_print_memref_0d<T>);

RAL_REGISTER_GPU_COPY_MEMREF_FUNC_0D(double);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC_0D(float);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC_0D(int32_t);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC_0D(int64_t);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC_0D(bool);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(double, 1);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(double, 2);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(double, 3);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(double, 4);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(double, 5);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(double, 6)
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(double, 7)
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(double, 8)
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(float, 1);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(float, 2);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(float, 3);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(float, 4);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(float, 5);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(float, 6);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(float, 7);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(float, 8);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(int32_t, 1);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(int32_t, 2);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(int32_t, 3);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(int32_t, 4);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(int32_t, 5);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(int32_t, 6);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(int32_t, 7);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(int32_t, 8);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(int64_t, 1);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(int64_t, 2);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(int64_t, 3);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(int64_t, 4);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(int64_t, 5);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(int64_t, 6);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(int64_t, 7);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(int64_t, 8);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(bool, 1);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(bool, 2);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(bool, 3);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(bool, 4);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(bool, 5);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(bool, 6);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(bool, 7);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(bool, 8);

#ifdef TAO_RAL_USE_STREAM_EXECUTOR
RAL_REGISTER_GPU_COPY_MEMREF_FUNC_0D(Eigen::half);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(Eigen::half, 1);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(Eigen::half, 2);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(Eigen::half, 3);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(Eigen::half, 4);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(Eigen::half, 5);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(Eigen::half, 6);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(Eigen::half, 7);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(Eigen::half, 8);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC_0D(int8_t);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(int8_t, 1);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(int8_t, 2);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(int8_t, 3);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(int8_t, 4);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(int8_t, 5);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(int8_t, 6);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(int8_t, 7);
RAL_REGISTER_GPU_COPY_MEMREF_FUNC(int8_t, 8);
#endif  // TAO_RAL_USE_STREAM_EXECUTOR

}  // namespace ral
}  // namespace tao

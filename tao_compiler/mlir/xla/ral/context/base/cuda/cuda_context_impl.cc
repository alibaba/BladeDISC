//===- cuda_context_impl.cc ----------------------===//
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

#include "mlir/xla/ral/context/base/cuda/cuda_context_impl.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mlir/xla/ral/context/common_context_impl.h"
#include "mlir/xla/ral/context/context_util.h"
#include "mlir/xla/ral/context/init_stream_executor.h"
#include "mlir/xla/ral/context/stream_executor_based_impl.h"
#include "mlir/xla/ral/ral_driver.h"
#include "mlir/xla/ral/ral_helper.h"
#include "mlir/xla/ral/ral_logging.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_types.h"

#if TENSORFLOW_USE_ROCM
#include "tensorflow/stream_executor/rocm/rocm_driver_wrapper.h"
#endif

namespace tao {
namespace ral {
namespace gpu {

using ::stream_executor::gpu::GpuDevicePtr;
using ::stream_executor::gpu::GpuFunctionHandle;
using ::stream_executor::gpu::GpuModuleHandle;
using ::stream_executor::gpu::GpuStatus;
#if TENSORFLOW_USE_ROCM
#define CUDA_SUCCESS hipSuccess
#endif

#define RETURN_VOID_ON_CUDA_ERROR(expr, msg) \
  {                                          \
    auto _cuda_error = (expr);               \
    if (_cuda_error != CUDA_SUCCESS) {       \
      TAO_VLOG(0) << "[[ ERROR ]]: " << msg; \
      return;                                \
    }                                        \
  }

#define RETURN_ON_CUDA_ERROR(expr, ret, msg) \
  {                                          \
    auto _cuda_error = (expr);               \
    if (_cuda_error != CUDA_SUCCESS) {       \
      TAO_VLOG(0) << "[[ ERROR ]]: " << msg; \
      return ret;                            \
    }                                        \
  }

const char* kRalBaseCudaContextState = "ral_base_cuda_context_state";

buffer_t gpu_alloc(size_t bytes) {
  GpuDevicePtr ptr;
#if TENSORFLOW_USE_ROCM
  RETURN_ON_CUDA_ERROR(stream_executor::wrap::hipMalloc(&ptr, bytes), nullptr,
                       "hipMalloc failed");
#else
  RETURN_ON_CUDA_ERROR(cuMemAlloc(&ptr, bytes), nullptr, "cuMemAlloc failed");
#endif
  return buffer_t(ptr);
}

void gpu_dealloc(buffer_t buffer) {
#if TENSORFLOW_USE_ROCM
  RETURN_VOID_ON_CUDA_ERROR(
      stream_executor::wrap::hipFree(absl::bit_cast<hipDeviceptr_t>(buffer)),
      "hipFree failed");
#else
  RETURN_VOID_ON_CUDA_ERROR(cuMemFree(CUdeviceptr(buffer)), "cuMemFree failed");
#endif
}

static int32_t reportErrorIfAny(GpuStatus result, ExecutionContext* ctx,
                                const char* where) {
  if (result != CUDA_SUCCESS) {
    std::ostringstream out;
    out << "GPU failed with " << result << " in " << where;
    ctx->signalError(Context::FAILURE, out.str());
  }
  return result;
}

static int32_t reportErrorIfAny(GpuStatus result, Context* ctx,
                                const char* where) {
  if (result != CUDA_SUCCESS) {
    std::ostringstream out;
    out << "GPU failed with " << result << " in " << where;
    ctx->signalError(Context::FAILURE, out.str());
  }
  return result;
}

struct BaseCudaContextState : public tao::ral::Context::Resource {
  std::mutex mu;

  GpuStreamHandle stream = nullptr;
  // map blob ptr -> loaded module
  std::map<void*, GpuModuleHandle> blobs;
  // map <blob ptr, kernel name> -> callable kernel
  std::map<std::pair<void*, std::string>, GpuFunctionHandle> kernels;

  std::shared_ptr<Allocator> gpu_allocator;
  bool cache_workspace_mem_across_execution;
#ifdef TAO_RAL_USE_STREAM_EXECUTOR
  ::stream_executor::Stream* se_stream;
#endif

  // buffers which are supposed to used across executions.
  std::unordered_set<const_buffer_t> device_persistent_buffers;

  void onExecutionFinish(ExecutionContext* ctx) override {
    std::lock_guard<std::mutex> lock(this->mu);
    if (!cache_workspace_mem_across_execution) {
      gpu_allocator->releaseAllFreeBuffers();
    }
  }

  void onContextFinish(Context* ctx) override {
#if TENSORFLOW_USE_ROCM
    reportErrorIfAny(stream_executor::wrap::hipStreamSynchronize(stream), ctx,
                     "StreamSync");
#else
    reportErrorIfAny(cuStreamSynchronize(stream), ctx, "StreamSync");
#endif
    for (const_buffer_t buffer : device_persistent_buffers) {
      gpu_allocator->dealloc(const_cast<buffer_t>(buffer));
    }
    for (auto& e : blobs) {
#if TENSORFLOW_USE_ROCM
      reportErrorIfAny(stream_executor::wrap::hipModuleUnload(e.second), ctx,
                       "ModuleUnload");
#else
      reportErrorIfAny(cuModuleUnload(e.second), ctx, "ModuleUnload");
#endif
    }
  }
};

std::unique_ptr<BaseContext> MakeBaseCudaContext(
    BaseContextOption& opt, ::tao::ral::cpu::BaseCpuContextOption& cpu_opt,
    ::tao::ral::gpu::BaseCudaContextOption& gpu_opt) {
  auto ctx = ::tao::ral::cpu::MakeBaseCpuContext(opt, cpu_opt);
  ctx->addDriver(::tao::ral::gpu::GPUDriver::name(),
                 std::unique_ptr<::tao::ral::gpu::GPUDriver>(
                     new ::tao::ral::gpu::GPUDriver(ctx.get())));

  ctx->getOrCreateResource(kRalBaseCudaContextState, [opt, gpu_opt]() {
    auto state = new BaseCudaContextState;
    state->stream = gpu_opt.stream;
    if (gpu_opt.gpu_allocator != nullptr) {
      state->gpu_allocator = gpu_opt.gpu_allocator;
    } else {
      state->gpu_allocator.reset(new InternalAllocator(gpu_alloc, gpu_dealloc));
    }
    state->cache_workspace_mem_across_execution =
        opt.cache_workspace_mem_across_execution;

#ifdef TAO_RAL_USE_STREAM_EXECUTOR
    if (gpu_opt.use_stream_executor) {
      state->se_stream = GetOrCreateDefaultCudaStreamExecutorStream(
          gpu_opt.device_ordinal, gpu_opt.stream);
      state->stream = (GpuStreamHandle)*state->se_stream->implementation()
                          ->GpuStreamMemberHack();
    }
#endif

    return state;
  });

  return ctx;
}

BaseCudaExecutionContext::BaseCudaExecutionContext(BaseContext* ctx)
    : BaseCpuExecutionContext(ctx) {}

BaseCudaExecutionContext::~BaseCudaExecutionContext() {}

void BaseCudaExecutionContext::setOutputDeleter(OutputBufferWrapper& output) {
  {
    if (synced) {
      synced = true;
      // TODO: pytorch may use multiple streams in this case stream
      // synchronization may still be necessary
      // reportErrorIfAny(cuStreamSynchronize(state->stream), this,
      // "StreamSync");
    }
    auto* state = getResource<BaseCudaContextState>(kRalBaseCudaContextState);
    std::lock_guard<std::mutex> lock(state->mu);
    const_buffer_t buffer = output.data();
    if (state->device_persistent_buffers.count(buffer)) {
      // This buffer is a pesistent buffer, thus no need to set a deleter.
      return;
    }

    auto dit = device_ptr_map.find(buffer);
    if (dit != device_ptr_map.end()) {
      if (--dit->second == 0) {
        static_cast<BaseOutputBufferWrapper*>(&output)->set_deleter(
            [state](buffer_t data) {
              std::lock_guard<std::mutex> lock(state->mu);
              state->gpu_allocator->dealloc(data);
            });
      }
      if (outputSharedOrder[output.data()] == 1) {
        output.markOwned();
      }
      return;
    }
  }
  BaseCpuExecutionContext::setOutputDeleter(output);
}

// ============================================================================
// ========================== gpu drvier api impl =============================
// ============================================================================

buffer_t ral_base_cuda_alloc(ExecutionContext* ctx, size_t bytes) {
  auto* state =
      ctx->getResource<BaseCudaContextState>(kRalBaseCudaContextState);
  auto exec_ctx = dynamic_cast<BaseCudaExecutionContext*>(ctx);

  std::lock_guard<std::mutex> lock(state->mu);
  TAO_VLOG(1) << "before ral_base_cuda_alloc alloc " << bytes;
  bytes = (bytes ? bytes : 1);
  void* ptr = state->gpu_allocator->alloc(bytes);
  TAO_VLOG(1) << "after ral_base_cuda_alloc with ptr=  " << ptr;
  exec_ctx->device_ptr_map.insert(std::make_pair(ptr, 1));
  return ptr;
}

buffer_t ral_base_cuda_alloc_persistent(ExecutionContext* ctx, size_t bytes) {
  auto* state =
      ctx->getResource<BaseCudaContextState>(kRalBaseCudaContextState);

  std::lock_guard<std::mutex> lock(state->mu);
  TAO_VLOG(1) << "before ral_base_cuda_alloc_persistent alloc " << bytes;
  bytes = (bytes ? bytes : 1);
  void* ptr = state->gpu_allocator->alloc(bytes);
  state->device_persistent_buffers.insert(ptr);
  TAO_VLOG(1) << "after ral_base_cuda_alloc_persistent with ptr=  " << ptr;
  return ptr;
}

void ral_base_cuda_dealloc(ExecutionContext* ctx, buffer_t buffer) {
  if (!buffer) {
    TAO_VLOG(1) << "ral_base_cuda_dealloc early return for nullptr";
    return;
  }

  auto* state =
      ctx->getResource<BaseCudaContextState>(kRalBaseCudaContextState);
  auto exec_ctx = dynamic_cast<BaseCudaExecutionContext*>(ctx);

  std::lock_guard<std::mutex> lock(state->mu);
  TAO_VLOG(1) << "before ral_base_cuda_dealloc with ptr = " << buffer;
  if (state->device_persistent_buffers.count(buffer)) return;
  auto it = exec_ctx->device_ptr_map.find(buffer);
  if (discEnableGlobalConstantStore()) {
    if (it == exec_ctx->device_ptr_map.end()) {
      TAO_VLOG(1) << "[[warning]] ignore buffer from other context\n";
      return;
    }
  }
  CHECK(it != exec_ctx->device_ptr_map.end());
  if (--it->second == 0) {
    state->gpu_allocator->dealloc(buffer);
    exec_ctx->device_ptr_map.erase(buffer);
    TAO_VLOG(1) << "delete buffer after ref-count becoming zero";
  }
  TAO_VLOG(1) << "after ral_base_cuda_dealloc with ptr =  " << buffer;
}

buffer_t ral_base_cuda_raw_alloc(Context* ctx, size_t bytes) {
  auto* state = static_cast<BaseCudaContextState*>(
      ctx->getOrCreateResource(kRalBaseCudaContextState, nullptr).get());
  TAO_VLOG(1) << "before ral_base_gpu_raw_alloc alloc " << bytes;
  buffer_t ptr = state->gpu_allocator->alloc(bytes);
  assert(ptr);
  TAO_VLOG(1) << "after ral_base_gpu_raw_alloc with ptr=  " << ptr;
  return ptr;
}

void ral_base_cuda_raw_dealloc(Context* ctx, buffer_t buffer) {
  if (!buffer) {
    TAO_VLOG(1) << "ral_base_cuda_raw_dealloc early return for nullptr";
    return;
  }

  auto* state = static_cast<BaseCudaContextState*>(
      ctx->getOrCreateResource(kRalBaseCudaContextState, nullptr).get());
  TAO_VLOG(1) << "before ral_base_gpu_raw_dealloc dealloc with ptr =  "
              << buffer;
  state->gpu_allocator->dealloc(buffer);
  TAO_VLOG(1) << "after ral_base_gpu_raw_dealloc with ptr =  " << buffer;
}

void ral_base_cuda_launch(ExecutionContext* ctx, void** blobs, size_t num_blobs,
                          const char* kernel_name, intptr_t gridX,
                          intptr_t gridY, intptr_t gridZ, intptr_t blockX,
                          intptr_t blockY, intptr_t blockZ,
                          int32_t smem,        /* sharedMemBytes */
                          void* stream_handle, /* stream */
                          int32_t num_args, void** params /* kernel params */) {
  // Skip if an empty launch
  if (!blockX || !blockY || !blockZ || !gridX || !gridY || !gridZ) {
    TAO_VLOG(1) << "skip launch kernel for empty tensor.";
    return;
  }

  if (TAO_VLOG_IS_ON(1)) {
    TAO_VLOG(1) << "ral_base_cuda_launch kernel@" << kernel_name << "[" << gridZ
                << ", " << gridY << ", " << gridX << "," << blockZ << ", "
                << blockY << "," << blockX << "] "
                << "shared_mem = " << smem << "\n";
  }

  auto* state =
      ctx->getResource<BaseCudaContextState>(kRalBaseCudaContextState);
  ::stream_executor::CudaComputeCapability cc = state->se_stream->parent()
                                                    ->GetDeviceDescription()
                                                    .cuda_compute_capability();

  // choose a proper blob
  void* blob = nullptr;
  if (num_blobs == 1) {
    blob = blobs[0];
  } else if (num_blobs > 1) {
    auto exec_ctx = dynamic_cast<BaseCudaExecutionContext*>(ctx);
    auto it = c_CC_INDEX_MAP.find(std::make_pair(cc.major, cc.minor));
    if (it == c_CC_INDEX_MAP.end() || it->second > num_blobs - 1) {
      // fallback with ptx
      TAO_VLOG(1) << "Use FatBinary with ptx, "
                  << "cc: " << cc.major << cc.minor
                  << ", num_blobs: " << num_blobs;
      blob = blobs[0];
    } else {
      TAO_VLOG(1) << "Use FatBinary with cubin of sm_" << cc.major << cc.minor;
      blob = blobs[it->second];
    }
  } else {
    TAO_VLOG(0) << "[[ ERROR ]]: Unexpected num_blobs: " << num_blobs;
    return;
  }

  // get or load kernel in blob
  GpuFunctionHandle function;
  GpuStreamHandle stream;
  {
    std::lock_guard<std::mutex> lock(state->mu);
    auto key = std::make_pair(blob, std::string(kernel_name));
    auto it = state->kernels.find(key);
    if (it == state->kernels.end()) {
      auto blob_it = state->blobs.find(blob);
      if (blob_it == state->blobs.end()) {
        GpuModuleHandle module;
#if TENSORFLOW_USE_ROCM
        reportErrorIfAny(
            stream_executor::wrap::hipModuleLoadData(&module, blob), ctx,
            "ModuleLoad");
#else
        reportErrorIfAny(cuModuleLoadFatBinary(&module, blob), ctx,
                         "ModuleLoad");
#endif
        blob_it = state->blobs.insert(std::make_pair(blob, module)).first;
      }
      auto module = blob_it->second;
      GpuFunctionHandle function;
#if TENSORFLOW_USE_ROCM
      reportErrorIfAny(stream_executor::wrap::hipModuleGetFunction(
                           &function, module, kernel_name),
                       ctx, "GetFunction");
#else
      reportErrorIfAny(cuModuleGetFunction(&function, module, kernel_name), ctx,
                       "GetFunction");
#endif
      it = state->kernels.insert(std::make_pair(key, function)).first;
    }
    function = it->second;
    stream = state->stream;
  }

#if TENSORFLOW_USE_ROCM
  reportErrorIfAny(stream_executor::wrap::hipModuleLaunchKernel(
                       function, gridX, gridY, gridZ, blockX, blockY, blockZ,
                       smem, stream, params, nullptr),
                   ctx, "LaunchKernel");
#else
  reportErrorIfAny(cuLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY,
                                  blockZ, smem, stream, params, nullptr),
                   ctx, "LaunchKernel");
#endif
}

stream_t ral_base_cuda_get_stream(ExecutionContext* ctx, int32_t stream_id) {
  if (stream_id < 0) {
    ctx->signalError(Context::FAILURE, "not a valid stream_id");
    return nullptr;
  }
  if (stream_id > 0) {
    ctx->signalError(Context::FAILURE, "multi-stream not supported");
    return nullptr;
  }

  intptr_t handle = stream_id;
  return (stream_t)(handle);
}

opaque_t ral_base_cuda_as_cu_stream(ExecutionContext* ctx, stream_t sidx) {
  auto* state =
      ctx->getResource<BaseCudaContextState>(kRalBaseCudaContextState);

  if ((intptr_t)(sidx) != 0) {
    ctx->signalError(Context::FAILURE, "not a valid stream idx");
    return nullptr;
  }

  return state->stream;
}

opaque_t ral_base_cuda_as_se_stream(ExecutionContext* ctx, stream_t sidx) {
#ifdef TAO_RAL_USE_STREAM_EXECUTOR
  auto* state =
      ctx->getResource<BaseCudaContextState>(kRalBaseCudaContextState);
  return state->se_stream;
#else
  return nullptr;
#endif
}

void ral_base_cuda_d2d(ExecutionContext* ctx, stream_t sidx, buffer_t from,
                       buffer_t to, size_t bytes) {
  if ((intptr_t)(sidx) != 0) {
    ctx->signalError(Context::FAILURE, "not a valid stream idx");
    return;
  }
  auto* state =
      ctx->getResource<BaseCudaContextState>(kRalBaseCudaContextState);
#if TENSORFLOW_USE_ROCM
  reportErrorIfAny(
      stream_executor::wrap::hipMemcpyDtoDAsync(
          (hipDeviceptr_t)to, (hipDeviceptr_t)from, bytes, state->stream),
      ctx, "ModuleLoad");
#else
  reportErrorIfAny(
      cuMemcpyAsync((CUdeviceptr)to, (CUdeviceptr)from, bytes, state->stream),
      ctx, "ModuleLoad");
#endif
}

void ral_base_cuda_sync_on_stream(ExecutionContext* ctx, stream_t sidx) {
  if ((intptr_t)(sidx) != 0) {
    ctx->signalError(Context::FAILURE, "not a valid stream idx");
    return;
  }

  auto* state =
      ctx->getResource<BaseCudaContextState>(kRalBaseCudaContextState);
#if TENSORFLOW_USE_ROCM
  reportErrorIfAny(stream_executor::wrap::hipStreamSynchronize(state->stream),
                   ctx, "StreamSync");
#else
  reportErrorIfAny(cuStreamSynchronize(state->stream), ctx, "StreamSync");
#endif
}

// ============================================================================
// ========================== basic kernel api impl
// =============================
// ============================================================================

void ral_base_cuda_bitcast_update_ref_count(ExecutionContext* ctx,
                                            buffer_t ptr) {
  auto ral_tf_ctx = dynamic_cast<BaseCudaExecutionContext*>(ctx);
  auto it = ral_tf_ctx->device_ptr_map.find(ptr);
  if (it != ral_tf_ctx->device_ptr_map.end()) {
    ++it->second;
  } else if (ral_tf_ctx->input_ptr_set.count(ptr)) {
    // We set ref count large than one since this buffer is borrowed from input
    // buffer, thus we do not want to relase it.
    ral_tf_ctx->device_ptr_map[ptr] = 2;
  } else {
    auto* state =
        ctx->getResource<BaseCudaContextState>(kRalBaseCudaContextState);
    const_buffer_t persistent_buffer = nullptr;
    {
      std::lock_guard<std::mutex> lock(state->mu);
      auto it = state->device_persistent_buffers.find(ptr);
      CHECK(it != state->device_persistent_buffers.end());
      persistent_buffer = *it;
    }
    CHECK(persistent_buffer != nullptr);
    // We set ref count large than one since this buffer is borrowed from
    // persistent_tensor, thus we do not want to relase it.
    ral_tf_ctx->device_ptr_map[persistent_buffer] = 2;
  }
}

template <typename T, int N>
::tao::ral::MemRefType<T, N> ral_base_cuda_bitcast(
    ExecutionContext* ctx, stream_t, ::tao::ral::MemRefType<T, N> input) {
  TAO_VLOG(1) << "ral_base_cuda_bitcast for " << N << "d\n";
  ::tao::ral::MemRefType<T, N> memref = input;

  if (memref.data) {
    ral_base_cuda_bitcast_update_ref_count(ctx, memref.data);
  }

  if (TAO_VLOG_IS_ON(1)) {
    ::tao::ral::print_memref(memref, "ral_base_cuda_bitcast");
  }
  return memref;
}

template <typename T>
::tao::ral::MemRefType<T, 0> ral_base_cuda_bitcast_0d(
    ExecutionContext* ctx, stream_t, ::tao::ral::MemRefType<T, 0> input) {
  TAO_VLOG(1) << "ral_base_cuda_bitcast for 0d";
  ::tao::ral::MemRefType<T, 0> memref = input;

  if (memref.data) {
    ral_base_cuda_bitcast_update_ref_count(ctx, memref.data);
  }

  if (TAO_VLOG_IS_ON(1)) {
    ::tao::ral::print_memref_0d<T>(memref, "ral_base_cuda_bitcast");
  }
  return memref;
}

template <typename T, int N, int M, typename P = int64_t>
::tao::ral::MemRefType<T, M> ral_base_cuda_bitcast(
    ExecutionContext* ctx, stream_t, ::tao::ral::MemRefType<T, N> input,
    ::tao::ral::MemRefType<P, 1> shape) {
  TAO_VLOG(1) << "ral_base_cuda_bitcast for " << M << "d\n";
  ::tao::ral::MemRefType<T, M> memref;

  memref.basePtr = input.basePtr;
  memref.data = input.data;
  memref.offset = 0;

  for (int i = 0; i < M; ++i) {
    memref.sizes[i] = shape.data[i];
  }

  memref.strides[M - 1] = 1;
  for (int i = M - 1; i > 0; --i) {
    memref.strides[i - 1] = memref.strides[i] * memref.sizes[i];
  }

  if (memref.data) {
    ral_base_cuda_bitcast_update_ref_count(ctx, memref.data);
  }

  if (TAO_VLOG_IS_ON(1)) {
    ::tao::ral::print_memref(memref, "ral_base_cuda_bitcast");
  }
  return memref;
}

template <typename T, int N, int M, typename P = int64_t>
::tao::ral::MemRefType<T, 0> ral_base_cuda_bitcast_0d(
    ExecutionContext* ctx, stream_t, ::tao::ral::MemRefType<T, N> input,
    ::tao::ral::MemRefType<P, 1> shape) {
  TAO_VLOG(1) << "ral_base_cuda_bitcast_0d for " << M << "d\n";
  ::tao::ral::MemRefType<T, 0> memref;

  memref.basePtr = input.basePtr;
  memref.data = input.data;
  memref.offset = 0;

  if (memref.data) {
    ral_base_cuda_bitcast_update_ref_count(ctx, memref.data);
  }

  if (TAO_VLOG_IS_ON(1)) {
    ::tao::ral::print_memref_0d(memref, "ral_base_cuda_bitcast_0d");
  }
  return memref;
}

void ral_base_cuda_h2d(ExecutionContext* ctx, void* stream_handle,
                       const void* h_src, buffer_t d_dst, size_t bytes) {
  auto* state =
      ctx->getResource<BaseCudaContextState>(kRalBaseCudaContextState);
#if TENSORFLOW_USE_ROCM
  reportErrorIfAny(
      stream_executor::wrap::hipMemcpyHtoDAsync(
          (GpuDevicePtr)d_dst, const_cast<void*>(h_src), bytes, state->stream),
      ctx, "cuMemcpyHtoDAsync");
#else
  reportErrorIfAny(
      cuMemcpyHtoDAsync((GpuDevicePtr)d_dst, h_src, bytes, state->stream), ctx,
      "cuMemcpyHtoDAsync");
#endif
}

void ral_base_cuda_d2h(ExecutionContext* ctx, void* stream_handle,
                       buffer_t d_src, buffer_t h_dst, size_t bytes) {
  auto* state =
      ctx->getResource<BaseCudaContextState>(kRalBaseCudaContextState);
#if TENSORFLOW_USE_ROCM
  reportErrorIfAny(
      stream_executor::wrap::hipMemcpyDtoHAsync(
          const_cast<void*>(h_dst), (GpuDevicePtr)d_src, bytes, state->stream),
      ctx, "cuMemcpyDtoHAsync");
#else
  reportErrorIfAny(
      cuMemcpyDtoHAsync(h_dst, (GpuDevicePtr)d_src, bytes, state->stream), ctx,
      "cuMemcpyDtoHAsync");
#endif
}

#define RAL_REGISTER_BITCAST_FUNC(T, N)                                      \
  TAO_RAL_API(tao::ral::kRalBitcast, "gpu", ral_base_cuda_bitcast<T, N>);    \
  TAO_RAL_API(tao::ral::kRalBitcast, "gpu", ral_base_cuda_bitcast<T, 0, N>); \
  TAO_RAL_API(tao::ral::kRalBitcast, "gpu", ral_base_cuda_bitcast<T, 1, N>); \
  TAO_RAL_API(tao::ral::kRalBitcast, "gpu", ral_base_cuda_bitcast<T, 2, N>); \
  TAO_RAL_API(tao::ral::kRalBitcast, "gpu", ral_base_cuda_bitcast<T, 3, N>); \
  TAO_RAL_API(tao::ral::kRalBitcast, "gpu", ral_base_cuda_bitcast<T, 4, N>); \
  TAO_RAL_API(tao::ral::kRalBitcast, "gpu", ral_base_cuda_bitcast<T, 5, N>); \
  TAO_RAL_API(tao::ral::kRalBitcast, "gpu", ral_base_cuda_bitcast<T, 6, N>); \
  TAO_RAL_API(tao::ral::kRalBitcast, "gpu", ral_base_cuda_bitcast<T, 7, N>); \
  TAO_RAL_API(tao::ral::kRalBitcast, "gpu", ral_base_cuda_bitcast<T, 8, N>);

#define RAL_REGISTER_BITCAST_FUNC_0D(T)                                        \
  TAO_RAL_API(tao::ral::kRalBitcast, "gpu", ral_base_cuda_bitcast_0d<T>);      \
  TAO_RAL_API(tao::ral::kRalBitcast, "gpu", ral_base_cuda_bitcast_0d<T, 0, 0>) \
  TAO_RAL_API(tao::ral::kRalBitcast, "gpu", ral_base_cuda_bitcast_0d<T, 1, 0>) \
  TAO_RAL_API(tao::ral::kRalBitcast, "gpu", ral_base_cuda_bitcast_0d<T, 2, 0>) \
  TAO_RAL_API(tao::ral::kRalBitcast, "gpu", ral_base_cuda_bitcast_0d<T, 3, 0>) \
  TAO_RAL_API(tao::ral::kRalBitcast, "gpu", ral_base_cuda_bitcast_0d<T, 4, 0>) \
  TAO_RAL_API(tao::ral::kRalBitcast, "gpu", ral_base_cuda_bitcast_0d<T, 5, 0>) \
  TAO_RAL_API(tao::ral::kRalBitcast, "gpu", ral_base_cuda_bitcast_0d<T, 6, 0>) \
  TAO_RAL_API(tao::ral::kRalBitcast, "gpu", ral_base_cuda_bitcast_0d<T, 7, 0>) \
  TAO_RAL_API(tao::ral::kRalBitcast, "gpu", ral_base_cuda_bitcast_0d<T, 8, 0>);

RAL_REGISTER_BITCAST_FUNC_0D(Eigen::half);
RAL_REGISTER_BITCAST_FUNC_0D(float);
RAL_REGISTER_BITCAST_FUNC_0D(double);
RAL_REGISTER_BITCAST_FUNC_0D(int32_t);
RAL_REGISTER_BITCAST_FUNC_0D(int64_t);
RAL_REGISTER_BITCAST_FUNC_0D(bool);
RAL_REGISTER_BITCAST_FUNC(Eigen::half, 1);
RAL_REGISTER_BITCAST_FUNC(Eigen::half, 2);
RAL_REGISTER_BITCAST_FUNC(Eigen::half, 3);
RAL_REGISTER_BITCAST_FUNC(Eigen::half, 4);
RAL_REGISTER_BITCAST_FUNC(Eigen::half, 5);
RAL_REGISTER_BITCAST_FUNC(Eigen::half, 6);
RAL_REGISTER_BITCAST_FUNC(Eigen::half, 7);
RAL_REGISTER_BITCAST_FUNC(Eigen::half, 8);
RAL_REGISTER_BITCAST_FUNC(float, 1);
RAL_REGISTER_BITCAST_FUNC(float, 2);
RAL_REGISTER_BITCAST_FUNC(float, 3);
RAL_REGISTER_BITCAST_FUNC(float, 4);
RAL_REGISTER_BITCAST_FUNC(float, 5);
RAL_REGISTER_BITCAST_FUNC(float, 6);
RAL_REGISTER_BITCAST_FUNC(float, 7);
RAL_REGISTER_BITCAST_FUNC(float, 8);
RAL_REGISTER_BITCAST_FUNC(double, 1);
RAL_REGISTER_BITCAST_FUNC(double, 2);
RAL_REGISTER_BITCAST_FUNC(double, 3);
RAL_REGISTER_BITCAST_FUNC(double, 4);
RAL_REGISTER_BITCAST_FUNC(double, 5);
RAL_REGISTER_BITCAST_FUNC(double, 6);
RAL_REGISTER_BITCAST_FUNC(double, 7);
RAL_REGISTER_BITCAST_FUNC(double, 8);
RAL_REGISTER_BITCAST_FUNC(int32_t, 1);
RAL_REGISTER_BITCAST_FUNC(int32_t, 2);
RAL_REGISTER_BITCAST_FUNC(int32_t, 3);
RAL_REGISTER_BITCAST_FUNC(int32_t, 4);
RAL_REGISTER_BITCAST_FUNC(int32_t, 5);
RAL_REGISTER_BITCAST_FUNC(int32_t, 6);
RAL_REGISTER_BITCAST_FUNC(int32_t, 7);
RAL_REGISTER_BITCAST_FUNC(int32_t, 8);
RAL_REGISTER_BITCAST_FUNC(int64_t, 1);
RAL_REGISTER_BITCAST_FUNC(int64_t, 2);
RAL_REGISTER_BITCAST_FUNC(int64_t, 3);
RAL_REGISTER_BITCAST_FUNC(int64_t, 4);
RAL_REGISTER_BITCAST_FUNC(int64_t, 5);
RAL_REGISTER_BITCAST_FUNC(int64_t, 6);
RAL_REGISTER_BITCAST_FUNC(int64_t, 7);
RAL_REGISTER_BITCAST_FUNC(int64_t, 8);
RAL_REGISTER_BITCAST_FUNC(bool, 1);
RAL_REGISTER_BITCAST_FUNC(bool, 2);
RAL_REGISTER_BITCAST_FUNC(bool, 3);
RAL_REGISTER_BITCAST_FUNC(bool, 4);
RAL_REGISTER_BITCAST_FUNC(bool, 5);
RAL_REGISTER_BITCAST_FUNC(bool, 6);
RAL_REGISTER_BITCAST_FUNC(bool, 7);
RAL_REGISTER_BITCAST_FUNC(bool, 8);

TAO_RAL_API(tao::ral::gpu::kRalGpuAlloc, "gpu", ral_base_cuda_alloc);
TAO_RAL_API(tao::ral::gpu::kRalGpuAllocPersistent, "gpu",
            ral_base_cuda_alloc_persistent);
TAO_RAL_API(tao::ral::gpu::kRalGpuDealloc, "gpu", ral_base_cuda_dealloc);
TAO_RAL_API(tao::ral::gpu::kRalGpuRawAlloc, "gpu", ral_base_cuda_raw_alloc);
TAO_RAL_API(tao::ral::gpu::kRalGpuRawDealloc, "gpu", ral_base_cuda_raw_dealloc);
TAO_RAL_API(tao::ral::gpu::kRalGpuLaunch, "gpu", ral_base_cuda_launch);
TAO_RAL_API(tao::ral::gpu::kRalGpuGetStream, "gpu", ral_base_cuda_get_stream);
TAO_RAL_API(tao::ral::gpu::kRalGpuAsCUStream, "gpu",
            ral_base_cuda_as_cu_stream);
TAO_RAL_API(tao::ral::gpu::kRalGpuAsSEStream, "gpu",
            ral_base_cuda_as_se_stream);
TAO_RAL_API(tao::ral::gpu::kRalGpuH2D, "gpu", ral_base_cuda_h2d);
TAO_RAL_API(tao::ral::gpu::kRalGpuD2H, "gpu", ral_base_cuda_d2h);
TAO_RAL_API(tao::ral::gpu::kRalGpuD2D, "gpu", ral_base_cuda_d2d);
TAO_RAL_API(tao::ral::gpu::kRalGpuSyncOnStream, "gpu",
            ral_base_cuda_sync_on_stream);

}  // namespace gpu
}  // namespace ral
}  // namespace tao

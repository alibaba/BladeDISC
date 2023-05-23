//===- gpu_driver.h ----------------------===//
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
// =============================================================================

#include "mlir/ral/device/gpu/gpu_driver.h"

#include <functional>

#include "mlir/ral/ral_context.h"
#include "mlir/ral/ral_helper.h"

namespace tao {
namespace ral {
namespace gpu {

const char* kRalGpuAlloc = "alloc";
const char* kRalGpuAllocPersistent = "ral_gpu_alloc_persistent";
const char* kRalGpuDealloc = "dealloc";
const char* kRalGpuRawAlloc = "raw_gpu_alloc";
const char* kRalGpuRawDealloc = "raw_gpu_dealloc";
const char* kRalGpuD2D = "d2d_";
const char* kRalGpuD2H = "d2h_";
const char* kRalGpuH2D = "h2d_";
const char* kRalGpuMemset = "ral_gpu_memset";
const char* kRalGpuLaunch = "ral_kernel_launch";
const char* kRalGpuGetStream = "ral_gpu_get_stream";
const char* kRalGpuSyncOnStream = "sync_on_stream";
const char* kRalGpuSyncAll = "ral_gpu_sync_all";
const char* kRalGpuAsCUStream = "ral_gpu_as_cu_stream";
const char* kRalGpuAsSEStream = "ral_gpu_as_se_stream";

struct GPUDriver::Impl {
  Context* context;
  using T = ExecutionContext*;
  std::function<buffer_t(T, size_t)> alloc;
  std::function<buffer_t(T, size_t)> alloc_persistent;
  std::function<void(T, buffer_t)> dealloc;
  std::function<buffer_t(Context*, size_t)> raw_alloc;
  std::function<void(Context*, buffer_t)> raw_dealloc;
  std::function<void(T, stream_t, buffer_t, buffer_t, size_t)> d2d;
  std::function<void(T, stream_t, buffer_t, buffer_t, size_t)> d2h;
  std::function<void(T, stream_t, const void*, buffer_t, size_t)> h2d;
  std::function<void(T, stream_t, buffer_t, int, size_t)> memset;
  std::function<stream_t(T, int)> getStream;
  std::function<int32_t(T, void**, size_t, const char*, intptr_t, intptr_t,
                        intptr_t, intptr_t, intptr_t, intptr_t, int32_t,
                        stream_t, void**)>
      launchKernel;
  std::function<void(T, stream_t)> syncOnStream;
  std::function<void(T)> syncAll;
  std::function<opaque_t(T, stream_t)> asCUStream;
  std::function<opaque_t(T, stream_t)> asSEStream;
};

GPUDriver::GPUDriver(Context* context) : impl_(new GPUDriver::Impl) {
  impl_->context = context;
  TAO_RAL_ASSIGN_TO_API_FUNC_WRAPPER(
      impl_->alloc,
      context->find(TaoRalApiFuncNameHelper<decltype(impl_->alloc)>::Invoke(
          std::string(kRalGpuAlloc) + "___gpu")));
  TAO_RAL_ASSIGN_TO_API_FUNC_WRAPPER(impl_->alloc_persistent,
                                     context->find(kRalGpuAllocPersistent));
  TAO_RAL_ASSIGN_TO_API_FUNC_WRAPPER(
      impl_->dealloc,
      context->find(TaoRalApiFuncNameHelper<decltype(impl_->dealloc)>::Invoke(
          std::string(kRalGpuDealloc) + "___gpu")));
  TAO_RAL_ASSIGN_TO_API_FUNC_WRAPPER(impl_->d2d, context->find(kRalGpuD2D));
  TAO_RAL_ASSIGN_TO_API_FUNC_WRAPPER(impl_->raw_alloc,
                                     context->find(kRalGpuRawAlloc));
  TAO_RAL_ASSIGN_TO_API_FUNC_WRAPPER(impl_->raw_dealloc,
                                     context->find(kRalGpuRawDealloc));

  TAO_RAL_ASSIGN_TO_API_FUNC_WRAPPER(impl_->d2h, context->find(kRalGpuD2H));
  TAO_RAL_ASSIGN_TO_API_FUNC_WRAPPER(impl_->h2d, context->find(kRalGpuH2D));
  TAO_RAL_ASSIGN_TO_API_FUNC_WRAPPER(impl_->memset,
                                     context->find(kRalGpuMemset));
  TAO_RAL_ASSIGN_TO_API_FUNC_WRAPPER(impl_->launchKernel,
                                     context->find(kRalGpuLaunch));
  TAO_RAL_ASSIGN_TO_API_FUNC_WRAPPER(impl_->getStream,
                                     context->find(kRalGpuGetStream));
  TAO_RAL_ASSIGN_TO_API_FUNC_WRAPPER(impl_->syncOnStream,
                                     context->find(kRalGpuSyncOnStream));
  TAO_RAL_ASSIGN_TO_API_FUNC_WRAPPER(impl_->syncAll,
                                     context->find(kRalGpuSyncAll));
  TAO_RAL_ASSIGN_TO_API_FUNC_WRAPPER(impl_->asCUStream,
                                     context->find(kRalGpuAsCUStream));
  TAO_RAL_ASSIGN_TO_API_FUNC_WRAPPER(impl_->asSEStream,
                                     context->find(kRalGpuAsSEStream));
}

GPUDriver::~GPUDriver() {}

/* static */ std::string GPUDriver::name() { return "GPUDriver"; }

buffer_t GPUDriver::alloc(ExecutionContext* ctx, size_t bytes) {
  if (!impl_->alloc) {
    impl_->context->signalError(Context::FAILURE,
                                kRalGpuAlloc + std::string(" not implemented"));
    return nullptr;
  }
  return impl_->alloc(ctx, bytes);
}

buffer_t GPUDriver::alloc_persistent(ExecutionContext* ctx, size_t bytes) {
  if (!impl_->alloc_persistent) {
    impl_->context->signalError(
        Context::FAILURE,
        kRalGpuAllocPersistent + std::string(" not implemented"));
    return nullptr;
  }
  return impl_->alloc_persistent(ctx, bytes);
}

void GPUDriver::dealloc(ExecutionContext* ctx, buffer_t buffer) {
  if (!impl_->dealloc) {
    impl_->context->signalError(
        Context::FAILURE, kRalGpuDealloc + std::string(" not implemented"));
    return;
  }
  impl_->dealloc(ctx, buffer);
}

buffer_t GPUDriver::raw_alloc(Context* ctx, size_t bytes) {
  if (!impl_->raw_alloc) {
    impl_->context->signalError(
        Context::FAILURE, kRalGpuRawAlloc + std::string(" not implemented"));
    return nullptr;
  }
  return impl_->raw_alloc(ctx, bytes);
}

void GPUDriver::raw_dealloc(Context* ctx, buffer_t buffer) {
  if (!impl_->raw_dealloc) {
    impl_->context->signalError(
        Context::FAILURE, kRalGpuRawDealloc + std::string(" not implemented"));
    return;
  }
  impl_->raw_dealloc(ctx, buffer);
}

void GPUDriver::d2d(ExecutionContext* ctx, stream_t handle, buffer_t from,
                    buffer_t to, size_t bytes) {
  if (!impl_->d2d) {
    impl_->context->signalError(Context::FAILURE,
                                kRalGpuD2D + std::string(" not implemented"));
    return;
  }
  impl_->d2d(ctx, handle, from, to, bytes);
}

void GPUDriver::d2h(ExecutionContext* ctx, stream_t handle, buffer_t from,
                    buffer_t to, size_t bytes) {
  if (!impl_->d2h) {
    impl_->context->signalError(Context::FAILURE,
                                kRalGpuD2H + std::string(" not implemented"));
    return;
  }
  impl_->d2h(ctx, handle, from, to, bytes);
}

void GPUDriver::h2d(ExecutionContext* ctx, stream_t handle, const void* from,
                    buffer_t to, size_t bytes) {
  if (!impl_->h2d) {
    impl_->context->signalError(Context::FAILURE,
                                kRalGpuH2D + std::string(" not implemented"));
    return;
  }
  impl_->h2d(ctx, handle, from, to, bytes);
}

void GPUDriver::memset(ExecutionContext* ctx, stream_t handle, buffer_t buffer,
                       int value, size_t count) {
  if (!impl_->memset) {
    impl_->context->signalError(
        Context::FAILURE, kRalGpuMemset + std::string(" not implemented"));
    return;
  }
  impl_->memset(ctx, handle, buffer, value, count);
}

stream_t GPUDriver::getStream(ExecutionContext* ctx, int idx) {
  if (!impl_->getStream) {
    impl_->context->signalError(
        Context::FAILURE, kRalGpuGetStream + std::string(" not implemented"));
    return nullptr;
  }
  return impl_->getStream(ctx, idx);
}

void GPUDriver::launchKernel(ExecutionContext* ctx, void** blobs,
                             size_t num_blobs, const char* kernel_name,
                             intptr_t gridX, intptr_t gridY, intptr_t gridZ,
                             intptr_t blockX, intptr_t blockY, intptr_t blockZ,
                             int32_t smem,    /* sharedMemBytes */
                             stream_t stream, /* stream */
                             void** params /* kernel params */) {
  if (!impl_->launchKernel) {
    impl_->context->signalError(
        Context::FAILURE, kRalGpuLaunch + std::string(" not implemented"));
    return;
  }
  impl_->launchKernel(ctx, blobs, num_blobs, kernel_name, gridX, gridY, gridZ,
                      blockX, blockY, blockZ, smem, stream, params);
}

void GPUDriver::syncOnStream(ExecutionContext* ctx, stream_t handle) {
  if (!impl_->syncOnStream) {
    impl_->context->signalError(
        Context::FAILURE,
        kRalGpuSyncOnStream + std::string(" not implemented"));
    return;
  }
  impl_->syncOnStream(ctx, handle);
}

void GPUDriver::syncAll(ExecutionContext* ctx) {
  if (!impl_->syncAll) {
    impl_->context->signalError(
        Context::FAILURE, kRalGpuSyncAll + std::string(" not implemented"));
    return;
  }
  impl_->syncAll(ctx);
}

opaque_t GPUDriver::asCUStream(ExecutionContext* ctx, stream_t handle) {
  if (!impl_->asCUStream) {
    impl_->context->signalError(
        Context::FAILURE, kRalGpuAsCUStream + std::string(" not implemented"));
    return nullptr;
  }
  return impl_->asCUStream(ctx, handle);
}

opaque_t GPUDriver::asSEStream(ExecutionContext* ctx, stream_t handle) {
  if (!impl_->asSEStream) {
    impl_->context->signalError(
        Context::FAILURE, kRalGpuAsSEStream + std::string(" not implemented"));
    return nullptr;
  }
  return impl_->asSEStream(ctx, handle);
}

}  // namespace gpu
}  // namespace ral
}  // namespace tao

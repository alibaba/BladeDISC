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

#ifndef RAL_GPU_GPU_DRIVER_H_
#define RAL_GPU_GPU_DRIVER_H_

#include <memory>
#include <string>

#include "mlir/xla/ral/ral_driver.h"

namespace tao {
namespace ral {

class Context;
class ExecutionContext;

namespace gpu {

using stream_t = void*;

extern const char* kRalGpuAlloc;
extern const char* kRalGpuAllocPersistent;
extern const char* kRalGpuDealloc;
extern const char* kRalGpuRawAlloc;
extern const char* kRalGpuRawDealloc;
extern const char* kRalGpuD2D;
extern const char* kRalGpuD2H;
extern const char* kRalGpuH2D;
extern const char* kRalGpuMemset;
extern const char* kRalGpuLaunch;
extern const char* kRalGpuGetStream;
extern const char* kRalGpuSyncOnStream;
extern const char* kRalGpuSyncAll;
extern const char* kRalGpuAsCUStream;
extern const char* kRalGpuAsSEStream;

// A core driver api set for GPU device.
class GPUDriver : public Driver {
 public:
  GPUDriver(Context* context);
  ~GPUDriver();

  static std::string name();

  // Allocator for device memory management
  // The allocator follows 'BFC' semantic during a single execution, which
  // enables the compiler to do even further optimization to hide launch
  // overhead.
  // `alloc` is used to alloc memory at execution level.
  // `alloc_persistent` is used to alloc memory across different executions.
  buffer_t alloc(ExecutionContext* ctx, size_t);
  buffer_t alloc_persistent(ExecutionContext* ctx, size_t);
  void dealloc(ExecutionContext* ctx, buffer_t);

  // Raw alloc & dealloc. The driver itself does not keep track of such buffers.
  // It's the responsibility of the user to manage the buffers correctly.
  buffer_t raw_alloc(Context* ctx, size_t);
  void raw_dealloc(Context* ctx, buffer_t);

  // memcpy between device and host or inside device
  void d2d(ExecutionContext* ctx, stream_t, buffer_t from, buffer_t to, size_t);
  void d2h(ExecutionContext* ctx, stream_t, buffer_t from, buffer_t to, size_t);
  void h2d(ExecutionContext* ctx, stream_t, const_buffer_t from, buffer_t to,
           size_t);
  void memset(ExecutionContext* ctx, stream_t, buffer_t, int, size_t);

  // Returns a stream hanlde identified by `idx`
  // This API uses implicit initialization semantic. Stream handle
  // is NOT own by the client.
  stream_t getStream(ExecutionContext* ctx, int idx);

  // Gpu launch abstraction.
  // Cubin isn't needed to be load by the client side. it's implicit
  // loaded (and managed) by RAL in case necessary.
  void launchKernel(ExecutionContext* ctx, void** blobs, size_t num_blobs,
                    const char* kernel_name, intptr_t gridX, intptr_t gridY,
                    intptr_t gridZ, intptr_t blockX, intptr_t blockY,
                    intptr_t blockZ, int32_t smem, /* sharedMemBytes */
                    stream_t stream,               /* stream */
                    void** params /* kernel params */);

  // sync on stream level
  void syncOnStream(ExecutionContext* ctx, stream_t);

  // sync on device level
  void syncAll(ExecutionContext* ctx);

  // Returns a platform-specific implementation, and nullptr if such
  // a conversion is not possible.
  // This is mainly used to simplify calling a library kernel, instead
  // of trying to wrapper every library call using driver api stream, to
  // make the driver api more clean and stable.
  opaque_t asCUStream(ExecutionContext* ctx, stream_t);
  opaque_t asSEStream(ExecutionContext* ctx, stream_t);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gpu
}  // namespace ral
}  // namespace tao

#endif  // RAL_GPU_GPU_DRIVER_H_

//===- cpu_driver.h ----------------------===//
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

#ifndef RAL_DEVICE_CPU_CPU_DRIVER_H_
#define RAL_DEVICE_CPU_CPU_DRIVER_H_

#include <memory>
#include <string>

#include "mlir/ral/ral_driver.h"
#include "mlir/ral/ral_helper.h"

namespace tao {
namespace ral {

class Context;
class ExecutionContext;

namespace cpu {

extern const char* kRalCpuAlloc;
extern const char* kRalCpuAllocPersistent;
extern const char* kRalCpuDealloc;
extern const char* kRalCpuRawAlloc;
extern const char* kRalCpuRawDealloc;
extern const char* kRalCpuMemcpy;
extern const char* kRalCpuMemset;
extern const char* kRalCpuLaunch;

using CpuLaunchDims = MemRefType<int64_t, 1>;

// A core driver api set for CPU device.
class CPUDriver : public Driver {
 public:
  CPUDriver(Context* context);
  ~CPUDriver();

  static std::string name();

  // Allocator for cpu memory management
  buffer_t alloc(ExecutionContext* ctx, size_t);
  buffer_t alloc_persistent(ExecutionContext* ctx, size_t);
  void dealloc(ExecutionContext* ctx, buffer_t);

  // Raw alloc & dealloc. The driver itself does not keep track of such buffers.
  // It's the responsibility of the user to manage the buffers correctly.
  buffer_t raw_alloc(Context* ctx, size_t);
  void raw_dealloc(Context* ctx, buffer_t);

  // memcpy & memset
  void memcpy(ExecutionContext* ctx, buffer_t from, buffer_t to, size_t bytes);
  void memset(ExecutionContext* ctx, buffer_t buffer, int value, size_t count);

  // cpu kernel launcher
  void launchKernel(ExecutionContext* ctx, const char* kernel_name,
                    CpuLaunchDims lowerBound, CpuLaunchDims upperBound,
                    CpuLaunchDims step, int64_t unitWorkloadSizeHint,
                    void* kernel, void** params /* kernel params */);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace cpu
}  // namespace ral
}  // namespace tao

#endif  // RAL_DEVICE_CPU_CPU_DRIVER_H_
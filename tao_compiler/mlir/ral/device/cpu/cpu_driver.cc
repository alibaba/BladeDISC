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
// ===========================================================================

#include "mlir/ral/device/cpu/cpu_driver.h"

#include <functional>

#include "mlir/ral/ral_context.h"
#include "mlir/ral/ral_helper.h"

namespace tao {
namespace ral {
namespace cpu {

const char* kRalCpuAlloc = "alloc";
const char* kRalCpuAllocPersistent = "ral_cpu_alloc_persistent";
const char* kRalCpuDealloc = "dealloc";
const char* kRalCpuRawAlloc = "raw_cpu_alloc";
const char* kRalCpuRawDealloc = "raw_cpu_dealloc";
const char* kRalCpuMemcpy = "ral_cpu_memcpy";
const char* kRalCpuMemset = "ral_cpu_memset";
const char* kRalCpuLaunch = "ral_kernel_launch";

struct CPUDriver::Impl {
  Context* context;
  using T = ExecutionContext*;
  std::function<buffer_t(T, size_t)> alloc;
  std::function<buffer_t(T, size_t)> alloc_persistent;
  std::function<buffer_t(Context*, size_t)> raw_alloc;
  std::function<void(Context*, buffer_t)> raw_dealloc;
  std::function<void(T, buffer_t)> dealloc;
  std::function<void(T, buffer_t, buffer_t, size_t)> memcpy;
  std::function<void(T, buffer_t, int, size_t)> memset;
  std::function<void(T, const char*, CpuLaunchDims, CpuLaunchDims,
                     CpuLaunchDims, int64_t, void*, void**)>
      launch;
};

CPUDriver::CPUDriver(Context* context) : impl_(new CPUDriver::Impl) {
  impl_->context = context;
  TAO_RAL_ASSIGN_TO_API_FUNC_WRAPPER(
      impl_->alloc,
      context->find(TaoRalApiFuncNameHelper<decltype(impl_->alloc)>::Invoke(
          std::string(kRalCpuAlloc) + "___cpu")));
  TAO_RAL_ASSIGN_TO_API_FUNC_WRAPPER(impl_->alloc_persistent,
                                     context->find(kRalCpuAllocPersistent));
  TAO_RAL_ASSIGN_TO_API_FUNC_WRAPPER(
      impl_->dealloc,
      context->find(TaoRalApiFuncNameHelper<decltype(impl_->dealloc)>::Invoke(
          std::string(kRalCpuDealloc) + "___cpu")));
  TAO_RAL_ASSIGN_TO_API_FUNC_WRAPPER(impl_->raw_alloc,
                                     context->find(kRalCpuRawAlloc));
  TAO_RAL_ASSIGN_TO_API_FUNC_WRAPPER(impl_->raw_dealloc,
                                     context->find(kRalCpuRawDealloc));
  TAO_RAL_ASSIGN_TO_API_FUNC_WRAPPER(impl_->memcpy,
                                     context->find(kRalCpuMemcpy));
  TAO_RAL_ASSIGN_TO_API_FUNC_WRAPPER(impl_->memset,
                                     context->find(kRalCpuMemset));
  TAO_RAL_ASSIGN_TO_API_FUNC_WRAPPER(impl_->launch,
                                     context->find(kRalCpuLaunch));
}

CPUDriver::~CPUDriver() {}

/* static */ std::string CPUDriver::name() { return "CPUDriver"; }

buffer_t CPUDriver::alloc(ExecutionContext* ctx, size_t bytes) {
  if (!impl_->alloc) {
    impl_->context->signalError(Context::FAILURE,
                                kRalCpuAlloc + std::string(" not implemented"));
    return nullptr;
  }
  return impl_->alloc(ctx, bytes);
}

buffer_t CPUDriver::alloc_persistent(ExecutionContext* ctx, size_t bytes) {
  if (!impl_->alloc_persistent) {
    impl_->context->signalError(
        Context::FAILURE,
        kRalCpuAllocPersistent + std::string(" not implemented"));
    return nullptr;
  }
  return impl_->alloc_persistent(ctx, bytes);
}

void CPUDriver::dealloc(ExecutionContext* ctx, buffer_t buffer) {
  if (!impl_->dealloc) {
    impl_->context->signalError(
        Context::FAILURE, kRalCpuDealloc + std::string(" not implemented"));
    return;
  }
  impl_->dealloc(ctx, buffer);
}

buffer_t CPUDriver::raw_alloc(Context* ctx, size_t bytes) {
  if (!impl_->raw_alloc) {
    impl_->context->signalError(
        Context::FAILURE, kRalCpuRawAlloc + std::string(" not implemented"));
    return nullptr;
  }
  return impl_->raw_alloc(ctx, bytes);
}

void CPUDriver::raw_dealloc(Context* ctx, buffer_t buffer) {
  if (!impl_->raw_dealloc) {
    impl_->context->signalError(
        Context::FAILURE, kRalCpuRawDealloc + std::string(" not implemented"));
    return;
  }
  impl_->raw_dealloc(ctx, buffer);
}

void CPUDriver::memcpy(ExecutionContext* ctx, buffer_t from, buffer_t to,
                       size_t bytes) {
  if (!impl_->memcpy) {
    impl_->context->signalError(
        Context::FAILURE, kRalCpuMemcpy + std::string(" not implemented"));
    return;
  }
  impl_->memcpy(ctx, from, to, bytes);
}

void CPUDriver::memset(ExecutionContext* ctx, buffer_t buffer, int value,
                       size_t count) {
  if (!impl_->memset) {
    impl_->context->signalError(
        Context::FAILURE, kRalCpuMemset + std::string(" not implemented"));
    return;
  }
  impl_->memset(ctx, buffer, value, count);
}

void CPUDriver::launchKernel(ExecutionContext* ctx, const char* kernel_name,
                             CpuLaunchDims lowerBound, CpuLaunchDims upperBound,
                             CpuLaunchDims step, int64_t unitWorkloadSizeHit,
                             void* kernel, void** params) {
  if (!impl_->launch) {
    impl_->context->signalError(
        Context::FAILURE, kRalCpuLaunch + std::string(" not implemented"));
    return;
  }
  impl_->launch(ctx, kernel_name, lowerBound, upperBound, step,
                unitWorkloadSizeHit, kernel, params);
}

}  // namespace cpu
}  // namespace ral
}  // namespace tao
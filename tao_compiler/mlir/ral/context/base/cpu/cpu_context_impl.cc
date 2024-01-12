//===- cpu_context_impl.cc ----------------------===//
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
#include "mlir/ral/context/base/cpu/cpu_context_impl.h"

#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mlir/ral/context/common_context_impl.h"
#include "mlir/ral/context/context_util.h"
#include "mlir/ral/ral_driver.h"
#include "mlir/ral/ral_helper.h"
#include "mlir/ral/ral_logging.h"

namespace tao {
namespace ral {
namespace cpu {

buffer_t cpu_alloc(size_t bytes) { return tao::ral::aligned_malloc(bytes); }

void cpu_dealloc(buffer_t buffer) { std::free(buffer); }

struct BaseCpuContextState : public tao::ral::Context::Resource {
  std::mutex mu;
  std::shared_ptr<Allocator> cpu_allocator;
  bool cache_workspace_mem_across_execution;

  // buffers which are supposed to used across executions.
  std::unordered_set<const_buffer_t> host_persistent_buffers;

  void onExecutionFinish(ExecutionContext* ctx) override {
    std::lock_guard<std::mutex> lock(this->mu);
    if (!cache_workspace_mem_across_execution) {
      cpu_allocator->releaseAllFreeBuffers();
    }
  }

  void onContextFinish(Context* ctx) override {
    for (const_buffer_t buffer : host_persistent_buffers) {
      cpu_allocator->dealloc(const_cast<buffer_t>(buffer));
    }
  }
};

const char* kRalBaseCpuContextState = "ral_base_cpu_context_state";

std::unique_ptr<BaseContext> MakeBaseCpuContext(BaseContextOption& opt,
                                                BaseCpuContextOption& cpu_opt) {
  std::unique_ptr<BaseContext> ctx(new BaseContext(opt));
  ctx->addDriver(::tao::ral::cpu::CPUDriver::name(),
                 std::unique_ptr<::tao::ral::cpu::CPUDriver>(
                     new ::tao::ral::cpu::CPUDriver(ctx.get())));

  ctx->getOrCreateResource(kRalBaseCpuContextState, [opt, cpu_opt]() {
    auto state = new BaseCpuContextState;
    if (cpu_opt.cpu_allocator != nullptr) {
      state->cpu_allocator = cpu_opt.cpu_allocator;
    } else {
      state->cpu_allocator.reset(new InternalAllocator(cpu_alloc, cpu_dealloc));
    }
    state->cache_workspace_mem_across_execution =
        opt.cache_workspace_mem_across_execution;

    return state;
  });

  return ctx;
}

BaseCpuExecutionContext::BaseCpuExecutionContext(BaseContext* ctx)
    : BaseExecutionContext(ctx) {}

BaseCpuExecutionContext::~BaseCpuExecutionContext() {}

void BaseCpuExecutionContext::setOutputDeleter(OutputBufferWrapper& output) {
  auto* state = getResource<BaseCpuContextState>(kRalBaseCpuContextState);
  std::lock_guard<std::mutex> lock(state->mu);
  const_buffer_t buffer = output.data();
  if (state->host_persistent_buffers.count(buffer)) {
    // This buffer is a pesistent buffer, thus no need to set a deleter.
    return;
  }

  auto hid = host_ptr_map.find(buffer);
  if (hid != host_ptr_map.end()) {
    if (--hid->second == 0) {
      static_cast<BaseOutputBufferWrapper*>(&output)->set_deleter(
          [state](buffer_t data) {
            std::lock_guard<std::mutex> lock(state->mu);
            state->cpu_allocator->dealloc(data);
          });
    }
    if (outputSharedOrder[output.data()] == 1) {
      output.markOwned();
    }
  } else {
    // Cpu buffer allocted by the compiler directly.
    // TODO: make compiler use ral to alloc cpu memory as well to remove this
    // part.
    static_cast<BaseOutputBufferWrapper*>(&output)->set_deleter(
        [state](buffer_t data) { cpu_dealloc(data); });
  }
}

// ============================================================================
// ========================== gpu drvier api impl =============================
// ============================================================================

buffer_t ral_base_cpu_alloc(ExecutionContext* ctx, size_t bytes) {
  auto* state = ctx->getResource<BaseCpuContextState>(kRalBaseCpuContextState);
  auto exec_ctx = dynamic_cast<BaseCpuExecutionContext*>(ctx);

  std::lock_guard<std::mutex> lock(state->mu);
  TAO_VLOG(1) << "before ral_base_cpu_alloc alloc " << bytes;
  bytes = (bytes ? bytes : 1);
  void* ptr = state->cpu_allocator->alloc(bytes);
  TAO_VLOG(1) << "after ral_base_cpu_alloc with ptr=  " << ptr;
  exec_ctx->host_ptr_map.insert(std::make_pair(ptr, 1));
  return ptr;
}

buffer_t ral_base_cpu_alloc_persistent(ExecutionContext* ctx, size_t bytes) {
  auto* state = ctx->getResource<BaseCpuContextState>(kRalBaseCpuContextState);

  std::lock_guard<std::mutex> lock(state->mu);
  TAO_VLOG(1) << "before ral_base_cpu_alloc_persistent alloc " << bytes;
  bytes = (bytes ? bytes : 1);
  void* ptr = state->cpu_allocator->alloc(bytes);
  state->host_persistent_buffers.insert(ptr);
  TAO_VLOG(1) << "after ral_base_cpu_alloc_persistent with ptr=  " << ptr;
  return ptr;
}

void ral_base_cpu_dealloc(ExecutionContext* ctx, buffer_t buffer) {
  if (!buffer) {
    TAO_VLOG(1) << "ral_base_cpu_dealloc early return for nullptr";
    return;
  }

  auto* state = ctx->getResource<BaseCpuContextState>(kRalBaseCpuContextState);
  auto exec_ctx = dynamic_cast<BaseCpuExecutionContext*>(ctx);

  std::lock_guard<std::mutex> lock(state->mu);

  // ignore persistent buffer.
  if (state->host_persistent_buffers.count(buffer)) {
    TAO_VLOG(1) << "ral_base_cpu_dealloc: ignore persistent buffer = "
                << buffer;
    return;
  }

  TAO_VLOG(1) << "before ral_base_cpu_dealloc with ptr = " << buffer;
  auto it = exec_ctx->host_ptr_map.find(buffer);
  if (it != exec_ctx->host_ptr_map.end() && --it->second == 0) {
    state->cpu_allocator->dealloc(buffer);
    exec_ctx->host_ptr_map.erase(it);
    TAO_VLOG(1) << "delete buffer after ref-count becoming zero";
  }
  TAO_VLOG(1) << "after ral_base_cpu_dealloc with ptr =  " << buffer;
}

buffer_t ral_base_cpu_raw_alloc(Context* ctx, size_t bytes) {
  auto* state = static_cast<BaseCpuContextState*>(
      ctx->getOrCreateResource(kRalBaseCpuContextState, nullptr).get());
  TAO_VLOG(1) << "before ral_base_raw_cpu_alloc alloc " << bytes;
  buffer_t ptr = state->cpu_allocator->alloc(bytes);
  assert(ptr);
  TAO_VLOG(1) << "after ral_base_raw_cpu_alloc with ptr=  " << ptr;
  return ptr;
}

void ral_base_cpu_raw_dealloc(Context* ctx, buffer_t buffer) {
  if (!buffer) {
    TAO_VLOG(1) << "ral_base_cpu_raw_dealloc early return for nullptr";
    return;
  }

  auto* state = static_cast<BaseCpuContextState*>(
      ctx->getOrCreateResource(kRalBaseCpuContextState, nullptr).get());
  TAO_VLOG(1) << "before ral_base_raw_cpu_dealloc dealloc with ptr =  "
              << buffer;
  state->cpu_allocator->dealloc(buffer);
  TAO_VLOG(1) << "after ral_base_raw_cpu_dealloc with ptr =  " << buffer;
}

// ============================================================================
// ========================== basic kernel api impl
// =============================
// ============================================================================
TAO_RAL_API(tao::ral::cpu::kRalCpuAlloc, "cpu", ral_base_cpu_alloc);
TAO_RAL_API(tao::ral::cpu::kRalCpuAllocPersistent, "cpu",
            ral_base_cpu_alloc_persistent);
TAO_RAL_API(tao::ral::cpu::kRalCpuDealloc, "cpu", ral_base_cpu_dealloc);
TAO_RAL_API(tao::ral::cpu::kRalCpuRawAlloc, "cpu", ral_base_cpu_raw_alloc);
TAO_RAL_API(tao::ral::cpu::kRalCpuRawDealloc, "cpu", ral_base_cpu_raw_dealloc);

buffer_t ral_base_cpu_alloc_test(ExecutionContext* ctx, size_t bytes) {
  TAO_VLOG(1) << "ral_tf_cpu_alloc_test bytes = " << bytes;
  buffer_t ptr = nullptr;
  if (bytes) {
    ptr = new char[bytes];
  }
  TAO_VLOG(1) << "ral_tf_cpu_alloc_test ptr = " << ptr;
  return ptr;
}
void ral_base_cpu_dealloc_test(ExecutionContext* ctx, buffer_t buffer) {
  TAO_VLOG(1) << "ral_tf_cpu_dealloc_test ptr = " << buffer;
  delete[] buffer;
}

TAO_RAL_API("tao_ral_cpu_alloc", "cpu", ral_base_cpu_alloc_test);
TAO_RAL_API("tao_ral_cpu_free", "cpu", ral_base_cpu_dealloc_test);

void ral_base_cpu_bitcast_update_ref_count(ExecutionContext* ctx,
                                           buffer_t ptr) {
  auto ral_tf_ctx = dynamic_cast<BaseCpuExecutionContext*>(ctx);
  auto it = ral_tf_ctx->host_ptr_map.find(ptr);
  if (it != ral_tf_ctx->host_ptr_map.end()) {
    ++it->second;
  } else if (ral_tf_ctx->input_ptr_set.count(ptr)) {
    // We set ref count large than one since this buffer is borrowed from input
    // buffer, thus we do not want to relase it.
    ral_tf_ctx->host_ptr_map[ptr] = 2;
  } else {
    auto* state =
        ctx->getResource<BaseCpuContextState>(kRalBaseCpuContextState);
    const_buffer_t persistent_buffer = nullptr;
    {
      std::lock_guard<std::mutex> lock(state->mu);
      auto it = state->host_persistent_buffers.find(ptr);
      assert(it != state->host_persistent_buffers.end());
      persistent_buffer = *it;
    }
    assert(persistent_buffer != nullptr);
    // We set ref count large than one since this buffer is borrowed from
    // persistent_tensor, thus we do not want to relase it.
    ral_tf_ctx->host_ptr_map[persistent_buffer] = 2;
  }
}

template <typename T, int N>
::tao::ral::MemRefType<T, N> ral_base_cpu_bitcast(
    ExecutionContext* ctx, void*, ::tao::ral::MemRefType<T, N> input) {
  TAO_VLOG(1) << "ral_base_cpu_bitcast for " << N << "d\n";
  ::tao::ral::MemRefType<T, N> memref = input;

  if (memref.data) {
    ral_base_cpu_bitcast_update_ref_count(ctx, memref.data);
  }

  if (TAO_VLOG_IS_ON(1)) {
    ::tao::ral::print_memref(memref, "ral_base_cpu_bitcast");
  }
  return memref;
}

template <typename T>
::tao::ral::MemRefType<T, 0> ral_base_cpu_bitcast_0d(
    ExecutionContext* ctx, void*, ::tao::ral::MemRefType<T, 0> input) {
  TAO_VLOG(1) << "ral_base_cpu_bitcast for 0d";
  ::tao::ral::MemRefType<T, 0> memref = input;

  if (memref.data) {
    ral_base_cpu_bitcast_update_ref_count(ctx, memref.data);
  }

  if (TAO_VLOG_IS_ON(1)) {
    ::tao::ral::print_memref_0d<T>(memref, "ral_base_cpu_bitcast");
  }
  return memref;
}

template <typename T, int N, int M, typename P = int64_t>
::tao::ral::MemRefType<T, M> ral_base_cpu_bitcast(
    ExecutionContext* ctx, void*, ::tao::ral::MemRefType<T, N> input,
    ::tao::ral::MemRefType<P, 1> shape) {
  TAO_VLOG(1) << "ral_base_cpu_bitcast for " << M << "d\n";
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
    ral_base_cpu_bitcast_update_ref_count(ctx, memref.data);
  }

  if (TAO_VLOG_IS_ON(1)) {
    ::tao::ral::print_memref(memref, "ral_base_cpu_bitcast");
  }
  return memref;
}

template <typename T, int N, int M, typename P = int64_t>
::tao::ral::MemRefType<T, 0> ral_base_cpu_bitcast_0d(
    ExecutionContext* ctx, void*, ::tao::ral::MemRefType<T, N> input,
    ::tao::ral::MemRefType<P, 1> shape) {
  TAO_VLOG(1) << "ral_base_cpu_bitcast_0d for " << M << "d\n";
  ::tao::ral::MemRefType<T, 0> memref;

  memref.basePtr = input.basePtr;
  memref.data = input.data;
  memref.offset = 0;

  if (memref.data) {
    ral_base_cpu_bitcast_update_ref_count(ctx, memref.data);
  }

  if (TAO_VLOG_IS_ON(1)) {
    ::tao::ral::print_memref_0d(memref, "ral_base_cpu_bitcast_0d");
  }
  return memref;
}

#define RAL_REGISTER_BITCAST_FUNC(T, N)                                     \
  TAO_RAL_API(tao::ral::kRalBitcast, "cpu", ral_base_cpu_bitcast<T, N>);    \
  TAO_RAL_API(tao::ral::kRalBitcast, "cpu", ral_base_cpu_bitcast<T, 0, N>); \
  TAO_RAL_API(tao::ral::kRalBitcast, "cpu", ral_base_cpu_bitcast<T, 1, N>); \
  TAO_RAL_API(tao::ral::kRalBitcast, "cpu", ral_base_cpu_bitcast<T, 2, N>); \
  TAO_RAL_API(tao::ral::kRalBitcast, "cpu", ral_base_cpu_bitcast<T, 3, N>); \
  TAO_RAL_API(tao::ral::kRalBitcast, "cpu", ral_base_cpu_bitcast<T, 4, N>); \
  TAO_RAL_API(tao::ral::kRalBitcast, "cpu", ral_base_cpu_bitcast<T, 5, N>); \
  TAO_RAL_API(tao::ral::kRalBitcast, "cpu", ral_base_cpu_bitcast<T, 6, N>); \
  TAO_RAL_API(tao::ral::kRalBitcast, "cpu", ral_base_cpu_bitcast<T, 7, N>); \
  TAO_RAL_API(tao::ral::kRalBitcast, "cpu", ral_base_cpu_bitcast<T, 8, N>);

#define RAL_REGISTER_BITCAST_FUNC_0D(T)                                       \
  TAO_RAL_API(tao::ral::kRalBitcast, "cpu", ral_base_cpu_bitcast_0d<T>);      \
  TAO_RAL_API(tao::ral::kRalBitcast, "cpu", ral_base_cpu_bitcast_0d<T, 0, 0>) \
  TAO_RAL_API(tao::ral::kRalBitcast, "cpu", ral_base_cpu_bitcast_0d<T, 1, 0>) \
  TAO_RAL_API(tao::ral::kRalBitcast, "cpu", ral_base_cpu_bitcast_0d<T, 2, 0>) \
  TAO_RAL_API(tao::ral::kRalBitcast, "cpu", ral_base_cpu_bitcast_0d<T, 3, 0>) \
  TAO_RAL_API(tao::ral::kRalBitcast, "cpu", ral_base_cpu_bitcast_0d<T, 4, 0>) \
  TAO_RAL_API(tao::ral::kRalBitcast, "cpu", ral_base_cpu_bitcast_0d<T, 5, 0>) \
  TAO_RAL_API(tao::ral::kRalBitcast, "cpu", ral_base_cpu_bitcast_0d<T, 6, 0>) \
  TAO_RAL_API(tao::ral::kRalBitcast, "cpu", ral_base_cpu_bitcast_0d<T, 7, 0>) \
  TAO_RAL_API(tao::ral::kRalBitcast, "cpu", ral_base_cpu_bitcast_0d<T, 8, 0>);

RAL_REGISTER_BITCAST_FUNC_0D(float);
RAL_REGISTER_BITCAST_FUNC_0D(double);
RAL_REGISTER_BITCAST_FUNC_0D(int32_t);
RAL_REGISTER_BITCAST_FUNC_0D(int64_t);
RAL_REGISTER_BITCAST_FUNC_0D(bool);
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

}  // namespace cpu
}  // namespace ral
}  // namespace tao

#ifdef TAO_RAL_USE_STREAM_EXECUTOR
#include "mlir/ral/context/stream_executor_based_impl.h"
namespace tao {
namespace ral {
namespace cpu {
RAL_REGISTER_BITCAST_FUNC_0D(Eigen::half);
RAL_REGISTER_BITCAST_FUNC(Eigen::half, 1);
RAL_REGISTER_BITCAST_FUNC(Eigen::half, 2);
RAL_REGISTER_BITCAST_FUNC(Eigen::half, 3);
RAL_REGISTER_BITCAST_FUNC(Eigen::half, 4);
RAL_REGISTER_BITCAST_FUNC(Eigen::half, 5);
RAL_REGISTER_BITCAST_FUNC(Eigen::half, 6);
RAL_REGISTER_BITCAST_FUNC(Eigen::half, 7);
RAL_REGISTER_BITCAST_FUNC(Eigen::half, 8);
}  // namespace cpu
}  // namespace ral
}  // namespace tao
#endif  // TAO_RAL_USE_STREAM_EXECUTOR

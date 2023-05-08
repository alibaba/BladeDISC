//===- tf_context_impl.h ----------------------===//
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

#include "mlir/ral/context/tensorflow/tf_context_impl.h"

#include <array>
#include <cstdlib>
#include <map>
#include <string>
#include <unordered_map>

#include "mlir/ral/context/common_context_impl.h"
#include "mlir/ral/context/context_util.h"
#include "mlir/ral/ral_driver.h"
#include "mlir/ral/ral_helper.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/stream_executor/device_description.h"

namespace se = stream_executor;

namespace tensorflow {

using buffer_t = ::tao::ral::buffer_t;
using opaque_t = ::tao::ral::opaque_t;
using Context = ::tao::ral::Context;
using ExecutionContext = ::tao::ral::ExecutionContext;
using stream_t = ::tao::ral::gpu::stream_t;

const char* kRalTFContextState = "ral_tf_context_state";

constexpr int kMaxEmptyTensorPointerWrapperSlotId = 1024 * 512 - 1;

struct RalTfExecutionContext::Impl {
  // TF op context
  OpKernelContext* op_ctx;
  // Allocated buffers are backed by TF tensors.
  // This makes sure that we do not need to copy output buffer
  // to tensorflow context.
  std::unordered_map<::tao::ral::buffer_t, std::pair<Tensor, int>> tensor_map;

  // Buffers that are the outputs of a cluster. These buffers are placed on cpu,
  // thus are allocated by te compiler directly using `malloc`. For such
  // buffers, we make a new copy and free these buffer at the end of the
  // execution context. Since one buffer may be used as output multiple times,
  // we need to track this to make sure that it only be freed once.
  std::unordered_set<buffer_t> to_free_host_output_buffers;

  // For each empty tensor, we return a pointer wrapper instread of returning
  // the nullptr directly, in order to identity each unique outstanding emtpy
  // tensor easily. The assumption behind this is that we never try to
  // dereference a nullptr, thus the actual value of the ptr does not matter.
  // Not using std::vector<bool> since it may be optimized to use bit instead of
  // byte to represent element data.
  int next_emtpy_tensor_wrapper_slot_id = 0;
  // pre-alloc buffer in order to ensure the pointer address is fixed.
  std::array<char, kMaxEmptyTensorPointerWrapperSlotId>
      empty_tensor_ptr_wrapper;
  std::vector<buffer_t> free_empty_tensor_wrapper_pool;

  std::mutex mu;
};

struct RalTFContextState : public ::tao::ral::Context::Resource {
  std::mutex mu;

  // map <blob ptr, kernel name> -> callable kernel
  std::map<std::pair<void*, std::string>, std::unique_ptr<::se::KernelBase>>
      kernels;

  // Allocated buffers are backed by TF tensors.
  // This makes sure that we do not need to copy output buffer
  // to tensorflow context.
  // This tensor store is used to back context-level allocations.
  std::unordered_map<::tao::ral::buffer_t, Tensor> tensor_map;

  // For each empty tensor, we return a pointer wrapper instread of returning
  // the nullptr directly, in order to identity each unique outstanding emtpy
  // tensor easily. The assumption behind this is that we never try to
  // dereference a nullptr, thus the actual value of the ptr does not matter.
  // Not using std::vector<bool> since it may be optimized to use bit instead of
  // byte to represent element data.
  int next_emtpy_tensor_wrapper_slot_id = 0;
  // pre-alloc buffer in order to ensure the pointer address is fixed.
  std::array<char, kMaxEmptyTensorPointerWrapperSlotId>
      empty_tensor_ptr_wrapper;
  std::vector<buffer_t> free_empty_tensor_wrapper_pool;

  void onExecutionFinish(ExecutionContext* ctx) override {
    auto ral_tf_ctx = dynamic_cast<RalTfExecutionContext*>(ctx);
    TAO_VLOG(2) << "ral_tf_ctx = " << ral_tf_ctx;
    // TOOD: re-visit if we want to support cross execution allocation.
    ral_tf_ctx->getImpl()->tensor_map.clear();
    for (auto buffer : ral_tf_ctx->getImpl()->to_free_host_output_buffers) {
      std::free(buffer);
    }
  }
};

RalTfContext::RalTfContext(const RalTfContextOptions& options) {
  // TODO: add a macro to control if we should enbale gpu.
  addDriver(::tao::ral::gpu::GPUDriver::name(),
            absl::make_unique<::tao::ral::gpu::GPUDriver>(this));
  addDriver(::tao::ral::cpu::CPUDriver::name(),
            absl::make_unique<::tao::ral::cpu::CPUDriver>(this));

  this->getOrCreateResource(kRalTFContextState, [options, this]() {
    auto state = new RalTFContextState;
    return state;
  });

  if (!options.metadata_file_path.empty()) {
    getOrCreateResource(::tao::ral::kRalGlobalConstantState, [options, this]() {
      auto state = new ::tao::ral::RalGlobalConstantState;

      // The metadata file is loaded once. The data will
      // be erased from metadata file once memcpy is done;
      if ((state->metadata = tao::ral::MetadataFile::loadFromFile(
               options.metadata_file_path))) {
        return state;
      } else {
        delete state;
        return (::tao::ral::RalGlobalConstantState*)nullptr;
      }
    });
  }
}

RalTfContext::~RalTfContext() {}

RalTfExecutionContext::RalTfExecutionContext(RalTfContext* ctx)
    : ExecutionContext(ctx), impl_(new Impl) {
  onExecutionStart();
}

RalTfExecutionContext::~RalTfExecutionContext() { onExecutionFinish(); }

void RalTfExecutionContext::setOpContext(OpKernelContext* ctx) {
  impl_->op_ctx = ctx;
}

OpKernelContext* RalTfExecutionContext::getOpContext() { return impl_->op_ctx; }

// ============================================================================
// ========================== gpu drvier api impl =============================
// ============================================================================

buffer_t WrapperIfEmpty(RalTfExecutionContext* ral_tf_ctx, buffer_t ptr) {
  if (ptr) return ptr;
  auto& free_wrapper_pool =
      ral_tf_ctx->getImpl()->free_empty_tensor_wrapper_pool;
  auto& next_slot_id = ral_tf_ctx->getImpl()->next_emtpy_tensor_wrapper_slot_id;
  auto& wrapper_pool = ral_tf_ctx->getImpl()->empty_tensor_ptr_wrapper;
  if (!free_wrapper_pool.empty()) {
    ptr = free_wrapper_pool.back();
    free_wrapper_pool.pop_back();
  } else {
    CHECK(next_slot_id < kMaxEmptyTensorPointerWrapperSlotId);
    ptr = &wrapper_pool[next_slot_id++];
  }
  TAO_VLOG(1) << "wrapper nullptr with: " << ptr;
  return ptr;
}

buffer_t WrapperIfEmpty(RalTFContextState* state, buffer_t ptr) {
  if (ptr) return ptr;
  auto& free_wrapper_pool = state->free_empty_tensor_wrapper_pool;
  auto& next_slot_id = state->next_emtpy_tensor_wrapper_slot_id;
  auto& wrapper_pool = state->empty_tensor_ptr_wrapper;
  if (!free_wrapper_pool.empty()) {
    ptr = free_wrapper_pool.back();
    free_wrapper_pool.pop_back();
  } else {
    CHECK(next_slot_id < kMaxEmptyTensorPointerWrapperSlotId);
    ptr = &wrapper_pool[next_slot_id++];
  }
  TAO_VLOG(1) << "wrapper nullptr with: " << ptr;
  return ptr;
}

bool IsEmptyBuffer(RalTfExecutionContext* ral_tf_ctx, buffer_t ptr) {
  auto& wrapper_pool = ral_tf_ctx->getImpl()->empty_tensor_ptr_wrapper;
  return (&wrapper_pool[0] <= ptr && ptr <= &wrapper_pool.back());
}

buffer_t ral_tf_gpu_alloc(ExecutionContext* ctx, size_t bytes) {
  auto ral_tf_ctx = dynamic_cast<RalTfExecutionContext*>(ctx);
  // We do not use `Allocator` directly since `tensorflow::Tensor`
  // does not has an public constructor to accept a
  // pre-allocated buffer (most recently code in upstream begins to
  // to support this) and we do not want to copy tensor for efficiency.
  Tensor tensor;
  auto s = ral_tf_ctx->getOpContext()->allocate_temp(
      DT_UINT8, {static_cast<int64_t>(bytes)}, &tensor);
  if (!s.ok()) {
    TAO_VLOG(1) << "ral_tf_gpu_alloc fail to alloc " << bytes
                << "B : " << s.error_message();
    ctx->signalError(Context::FAILURE, "fail to alloc: " + s.error_message());
    return nullptr;
  }
  auto ptr = (buffer_t)tensor.tensor_data().data();
  // Wrapper a empty tensor to give it a unique address
  ptr = WrapperIfEmpty(ral_tf_ctx, ptr);
  ral_tf_ctx->getImpl()->tensor_map[ptr] = std::make_pair(tensor, 1);
  TAO_VLOG(1) << "ral_tf_gpu_alloc (bytes):" << bytes << " ptr: " << ptr;
  return ptr;
}

buffer_t ral_tf_gpu_alloc_persistent(ExecutionContext* ctx, size_t bytes) {
  auto ral_tf_ctx = dynamic_cast<RalTfExecutionContext*>(ctx);
  auto* state = ctx->getResource<RalTFContextState>(kRalTFContextState);
  // We do not use `Allocator` directly since `tensorflow::Tensor`
  // does not has an public constructor to accept a
  // pre-allocated buffer (most recently code in upstream begins to
  // to support this) and we do not want to copy tensor for efficiency.
  Tensor tensor;
  ral_tf_ctx->getOpContext()->allocate_temp(
      DT_UINT8, {static_cast<int64_t>(bytes)}, &tensor);
  auto ptr = (buffer_t)tensor.tensor_data().data();

  {
    std::lock_guard<std::mutex> lock(state->mu);
    ptr = WrapperIfEmpty(state, ptr);
    state->tensor_map[ptr] = tensor;
  }
  TAO_VLOG(1) << "ral_tf_gpu_alloc_persistent (bytes):" << bytes
              << " ptr: " << ptr;
  return ptr;
}

void ral_tf_gpu_dealloc(ExecutionContext* ctx, buffer_t buffer) {
  auto ral_tf_ctx = dynamic_cast<RalTfExecutionContext*>(ctx);
  TAO_VLOG(1) << "ral_tf_gpu_dealloc (buffer):" << buffer;

  auto it = ral_tf_ctx->getImpl()->tensor_map.find(buffer);
  if (it != ral_tf_ctx->getImpl()->tensor_map.end()) {
    if (--it->second.second == 0) {
      ral_tf_ctx->getImpl()->tensor_map.erase(it);
      if (IsEmptyBuffer(ral_tf_ctx, buffer)) {
        ral_tf_ctx->getImpl()->free_empty_tensor_wrapper_pool.push_back(buffer);
      }
    }
  }
}

buffer_t ral_tf_cpu_alloc(ExecutionContext* ctx, size_t bytes) {
  auto ral_tf_ctx = dynamic_cast<RalTfExecutionContext*>(ctx);
  // We do not use `Allocator` directly since `tensorflow::Tensor`
  // does not has an public constructor to accept a
  // pre-allocated buffer (most recently code in upstream begins to
  // to support this) and we do not want to copy tensor for efficiency.
  Tensor tensor;
  AllocatorAttributes alloc_attr;
  alloc_attr.set_on_host(true);
  ral_tf_ctx->getOpContext()->allocate_temp(
      DT_UINT8, {static_cast<int64_t>(bytes)}, &tensor, alloc_attr);
  auto ptr = (buffer_t)tensor.tensor_data().data();
  {
    std::lock_guard<std::mutex> lock(ral_tf_ctx->getImpl()->mu);
    // Wrapper a empty tensor to give it a unique address
    ptr = WrapperIfEmpty(ral_tf_ctx, ptr);
    ral_tf_ctx->getImpl()->tensor_map[ptr] = std::make_pair(tensor, 1);
  }
  TAO_VLOG(1) << "ral_tf_cpu_alloc (bytes):" << bytes << " ptr: " << ptr;
  return ptr;
}

buffer_t ral_tf_cpu_alloc_persistent(ExecutionContext* ctx, size_t bytes) {
  auto ral_tf_ctx = dynamic_cast<RalTfExecutionContext*>(ctx);
  auto* state = ctx->getResource<RalTFContextState>(kRalTFContextState);
  // We do not use `Allocator` directly since `tensorflow::Tensor`
  // does not has an public constructor to accept a
  // pre-allocated buffer (most recently code in upstream begins to
  // to support this) and we do not want to copy tensor for efficiency.
  Tensor tensor;
  AllocatorAttributes alloc_attr;
  alloc_attr.set_on_host(true);
  ral_tf_ctx->getOpContext()->allocate_temp(
      DT_UINT8, {static_cast<int64_t>(bytes)}, &tensor, alloc_attr);
  auto ptr = (buffer_t)tensor.tensor_data().data();

  {
    std::lock_guard<std::mutex> lock(state->mu);
    ptr = WrapperIfEmpty(state, ptr);
    state->tensor_map[ptr] = tensor;
  }
  TAO_VLOG(1) << "ral_tf_cpu_alloc_persistent (bytes):" << bytes
              << " ptr: " << ptr;
  return ptr;
}

void ral_tf_cpu_dealloc(ExecutionContext* ctx, buffer_t buffer) {
  auto ral_tf_ctx = dynamic_cast<RalTfExecutionContext*>(ctx);

  std::lock_guard<std::mutex> lock(ral_tf_ctx->getImpl()->mu);
  auto it = ral_tf_ctx->getImpl()->tensor_map.find(buffer);
  if (it != ral_tf_ctx->getImpl()->tensor_map.end()) {
    if (--it->second.second == 0) {
      ral_tf_ctx->getImpl()->tensor_map.erase(it);
      if (IsEmptyBuffer(ral_tf_ctx, buffer)) {
        ral_tf_ctx->getImpl()->free_empty_tensor_wrapper_pool.push_back(buffer);
      }
    }
  }
}

buffer_t ral_tf_cpu_alloc_test(ExecutionContext* ctx, size_t bytes) {
  TAO_VLOG(1) << "ral_tf_cpu_alloc_test bytes = " << bytes;
  buffer_t ptr = nullptr;
  if (bytes) {
    ptr = std::malloc(bytes);
  }
  TAO_VLOG(1) << "ral_tf_cpu_alloc_test ptr = " << ptr;
  return ptr;
}

void ral_tf_cpu_dealloc_test(ExecutionContext* ctx, buffer_t buffer) {
  TAO_VLOG(1) << "ral_tf_cpu_dealloc_test ptr = " << buffer;
  if (buffer) {
    std::free(buffer);
  }
}

class RalTfKernelArgsArrayBase : public se::KernelArgsArrayBase {
 public:
  RalTfKernelArgsArrayBase(void** params, size_t num_args)
      : params_(params),
        num_args_(num_args),
        // TODO(disc): currently backend driver does not use this sizes array,
        // thus we set an default value here. Re-visit this if a backend really
        // use these fields.
        argument_sizes_(num_args, sizeof(void*)) {}

  // Gets the number of arguments added so far, including shared memory
  // arguments.
  virtual size_t number_of_arguments() const override { return num_args_; }

// Gets the total number of shared memory bytes added so far.
// TODO(disc): re-visit this once we need to explicitly set shared memory size.
#if (TF_MAJOR_VERSION < 2 || (TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION < 6))
  uint64
#else
  uint64_t
#endif
  number_of_shared_bytes() const override {
    return 0;
  }

  // Gets the list of argument addresses.
  se::port::ArraySlice<const void*> argument_addresses() const override {
    return se::port::ArraySlice<const void*>(params_, num_args_);
  }

  // Gets an iterator to the arguments in the array.
  se::KernelArgIterator arg_iterator() const override {
    return se::KernelArgIterator(num_args_, 0, params_, argument_sizes_.data(),
                                 /*shared_memory_bytes*/ nullptr, nullptr);
  }

 private:
  void** params_;
  size_t num_args_;
  std::vector<size_t> argument_sizes_;
};

void ral_tf_gpu_launch(ExecutionContext* ctx, void** blobs, size_t num_blobs,
                       const char* kernel_name, intptr_t gridX, intptr_t gridY,
                       intptr_t gridZ, intptr_t blockX, intptr_t blockY,
                       intptr_t blockZ, int32_t smem, /* sharedMemBytes */
                       void* stream_handle,           /* stream */
                       int32_t num_args, void** params) /* kernel params */ {
  auto* state = ctx->getResource<RalTFContextState>(kRalTFContextState);
  auto ral_tf_ctx = dynamic_cast<RalTfExecutionContext*>(ctx);
  TAO_VLOG(1) << "launch kernel: " << kernel_name << " with (" << gridX << ", "
              << gridY << ", " << gridZ << ") blocks, (" << blockX << ", "
              << blockY << ", " << blockZ << ") threads";

  // Skip if an empty launch
  if (!blockX || !blockY || !blockZ || !gridX || !gridY || !gridZ) {
    TAO_VLOG(1) << "skip launch kernel for empty tensor.";
    return;
  }

  // get or load kernel in fatbin
  se::Stream* stream = nullptr;
  se::StreamExecutor* executor = nullptr;
  se::KernelBase* kernel_ptr = nullptr;
  std::string namedeb;
  {
    std::lock_guard<std::mutex> lock(state->mu);
    stream = ral_tf_ctx->getOpContext()->op_device_context()->stream();
    CHECK(stream != nullptr);
    executor = stream->parent();

    // choose a proper blob
    void* blob = nullptr;
    if (num_blobs == 1) {
      blob = blobs[0];
    } else if (num_blobs > 1) {
      // these codes are reserved to support AOT with tensorflow in future
      auto exec_ctx = dynamic_cast<RalTfExecutionContext*>(ctx);
      int cc_major;
      int cc_minor;
#if defined(IS_PAI_TF) || TF_MAJOR_VERSION < 2 || \
    (TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION < 6)
      if (!executor->GetDeviceDescription().cuda_compute_capability(
              &cc_major, &cc_minor)) {
        VLOG(0) << "[[ ERROR ]]: failed to get cuda_compute_capability";
        return;
      }
#else
      ::se::CudaComputeCapability cc =
          executor->GetDeviceDescription().cuda_compute_capability();
#endif
      auto it =
          tao::ral::c_CC_INDEX_MAP.find(std::make_pair(cc_major, cc_minor));
      if (it == tao::ral::c_CC_INDEX_MAP.end() || it->second > num_blobs - 1) {
        // fallback with ptx blob
        VLOG(2) << "Use FatBinary with ptx";
        blob = blobs[0];
      } else {
        VLOG(2) << "Use FatBinary with cubin of sm_" << cc_major << cc_minor;
        blob = blobs[it->second];
      }
    } else {
      VLOG(0) << "[[ ERROR ]]: Unexpected num_blobs: " << num_blobs;
      return;
    }

    auto key = std::make_pair(blob, std::string(kernel_name));
    auto it = state->kernels.find(key);
    namedeb = key.second;
    if (it == state->kernels.end()) {
      se::MultiKernelLoaderSpec spec(num_args);
      spec.AddCudaCubinInMemory((char*)blob, (char*)kernel_name);
      std::unique_ptr<se::KernelBase> kernel(new se::KernelBase(executor));
      auto status = ral_to_bool(executor->GetKernel(spec, kernel.get()));
      if (!status) {
        VLOG(0) << "unable to load kernel";
        ctx->signalError(Context::FAILURE, "fail to find kernel " + key.second);
        return;
      }
      it = state->kernels.insert(std::make_pair(key, std::move(kernel))).first;
    }
    kernel_ptr = it->second.get();
  }

  RalTfKernelArgsArrayBase kernel_args(params, num_args);
  if (VLOG_IS_ON(2)) {
    VLOG(2) << "kernel is " << namedeb;
    se::KernelArgIterator iter = kernel_args.arg_iterator();
    while (iter.has_next()) {
      se::KernelArg arg = iter.next();
      VLOG(2) << "*(arg.address):"
              << reinterpret_cast<uint64_t>(
                     *static_cast<const uint64_t*>(arg.address));
    }
  }

  auto status = ral_to_bool(executor->Launch(
      stream, se::ThreadDim(blockX, blockY, blockZ),
      se::BlockDim(gridX, gridY, gridZ), *kernel_ptr, kernel_args));
  if (!status) {
    VLOG(0) << "unable to execute kernel";
    ctx->signalError(Context::FAILURE,
                     "fail to launch kernel " + std::string(kernel_name));
    return;
  }

  if (tao::ral::isDebugMode()) {
    auto ral_tf_ctx = dynamic_cast<RalTfExecutionContext*>(ctx);
    auto s = ral_tf_ctx->getOpContext()
                 ->op_device_context()
                 ->stream()
                 ->BlockHostUntilDone();
    if (!s.ok()) {
      VLOG(0) << "failed to launch: " << s.error_message();
      return;
    }
  }
}

stream_t ral_tf_gpu_get_stream(ExecutionContext* ctx, int32_t stream_id) {
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

opaque_t ral_tf_gpu_as_cu_stream(ExecutionContext* ctx, stream_t sidx) {
  if ((intptr_t)(sidx) != 0) {
    ctx->signalError(Context::FAILURE, "not a valid stream idx");
    return nullptr;
  }

  auto ral_tf_ctx = dynamic_cast<RalTfExecutionContext*>(ctx);
  return *ral_tf_ctx->getOpContext()
              ->op_device_context()
              ->stream()
              ->implementation()
              ->GpuStreamMemberHack();
}

opaque_t ral_tf_gpu_as_se_stream(ExecutionContext* ctx, stream_t sidx) {
  if ((intptr_t)(sidx) != 0) {
    ctx->signalError(Context::FAILURE, "not a valid stream idx");
    return nullptr;
  }

  auto ral_tf_ctx = dynamic_cast<RalTfExecutionContext*>(ctx);

  // We don't support multi-stream currently, thus the real value doesn't
  // matter.
  // TODO: this should be re-visited if we want to support multi-stream.
  return ral_tf_ctx->getOpContext()->op_device_context()->stream();
}

void ral_tf_gpu_d2d(ExecutionContext* ctx, stream_t sid, buffer_t from,
                    buffer_t to, size_t bytes) {
  if (sid != 0) {
    ctx->signalError(Context::FAILURE, "multi-stream is not supported ATM");
    return;
  }

  auto ral_tf_ctx = dynamic_cast<RalTfExecutionContext*>(ctx);

  se::DeviceMemoryBase dst(to);
  se::DeviceMemoryBase src(from);

  ral_tf_ctx->getOpContext()->op_device_context()->stream()->ThenMemcpy(
      &dst, src, bytes);
  return;
}

void ral_tf_gpu_h2d(ExecutionContext* ctx, stream_t sid, const void* from,
                    buffer_t to, size_t bytes) {
  if (sid != 0) {
    ctx->signalError(Context::FAILURE, "multi-stream is not supported ATM");
    return;
  }

  auto ral_tf_ctx = dynamic_cast<RalTfExecutionContext*>(ctx);

  se::DeviceMemoryBase dst(to);

  ral_tf_ctx->getOpContext()->op_device_context()->stream()->ThenMemcpy(
      &dst, from, bytes);
  return;
}

void ral_tf_gpu_d2h(ExecutionContext* ctx, stream_t sid, buffer_t from,
                    buffer_t to, size_t bytes) {
  if (sid != 0) {
    ctx->signalError(Context::FAILURE, "multi-stream is not supported ATM");
    return;
  }

  auto ral_tf_ctx = dynamic_cast<RalTfExecutionContext*>(ctx);

  se::DeviceMemoryBase src(from);

  ral_tf_ctx->getOpContext()->op_device_context()->stream()->ThenMemcpy(to, src,
                                                                        bytes);
  return;
}

void ral_tf_gpu_sync_on_stream(ExecutionContext* ctx, stream_t sidx) {
  if ((intptr_t)(sidx) != 0) {
    ctx->signalError(Context::FAILURE, "not a valid stream idx");
    return;
  }

  auto ral_tf_ctx = dynamic_cast<RalTfExecutionContext*>(ctx);
  ral_tf_ctx->getOpContext()
      ->op_device_context()
      ->stream()
      ->BlockHostUntilDone();
}

template <typename T, int N>
::tao::ral::MemRefType<T, N> ral_tf_recv_input(ExecutionContext* ctx,
                                               int64_t input_idx) {
  TAO_VLOG(2) << "ral_tf_recv_input " << input_idx << " for " << N << "d\n";
  ::tao::ral::MemRefType<T, N> memref;

  auto ral_tf_ctx = dynamic_cast<RalTfExecutionContext*>(ctx);

  Tensor tensor = ral_tf_ctx->getOpContext()->input(input_idx);
  memref.basePtr = (T*)(tensor.tensor_data().data());
  memref.data = memref.basePtr;
  memref.offset = 0;

  ral_tf_ctx->getImpl()->tensor_map[memref.data].first = tensor;
  ++ral_tf_ctx->getImpl()->tensor_map[memref.data].second;

  for (int i = 0; i < N; ++i) {
    memref.sizes[i] = tensor.dim_size(i);
  }

  memref.strides[N - 1] = 1;
  for (int i = N - 1; i > 0; --i) {
    memref.strides[i - 1] = memref.strides[i] * memref.sizes[i];
  }

  if (TAO_VLOG_IS_ON(1)) {
    ::tao::ral::print_memref(memref, "input");
  }
  return memref;
}

template <typename T>
::tao::ral::MemRefType<T, 0> ral_tf_recv_input_0d(ExecutionContext* ctx,
                                                  int64_t input_idx) {
  TAO_VLOG(2) << "ral_tf_recv_input " << input_idx << " for 0d";
  ::tao::ral::MemRefType<T, 0> memref;

  auto ral_tf_ctx = dynamic_cast<RalTfExecutionContext*>(ctx);

  Tensor tensor = ral_tf_ctx->getOpContext()->input(input_idx);
  memref.basePtr = (T*)(tensor.tensor_data().data());
  memref.data = memref.basePtr;
  memref.offset = 0;

  ral_tf_ctx->getImpl()->tensor_map[memref.data].first = tensor;
  ++ral_tf_ctx->getImpl()->tensor_map[memref.data].second;

  if (TAO_VLOG_IS_ON(1)) {
    ::tao::ral::print_memref_0d<T>(memref, "input");
  }
  return memref;
}

void ral_tf_bitcast_update_ref_count(ExecutionContext* ctx, buffer_t ptr) {
  TAO_VLOG(1) << "ral_tf_bitcast_update_ref_count for ptr = " << ptr;
  auto ral_tf_ctx = dynamic_cast<RalTfExecutionContext*>(ctx);
  auto it = ral_tf_ctx->getImpl()->tensor_map.find(ptr);
  if (it != ral_tf_ctx->getImpl()->tensor_map.end()) {
    ++it->second.second;
  } else {
    auto* state = ctx->getResource<RalTFContextState>(kRalTFContextState);
    Tensor* persistent_tensor = nullptr;
    {
      std::lock_guard<std::mutex> lock(state->mu);
      auto it = state->tensor_map.find(ptr);
      CHECK(it != state->tensor_map.end());
      persistent_tensor = &it->second;
    }
    CHECK(persistent_tensor != nullptr);
    // We set ref count large than one since this buffer is borrowed from
    // persistent_tensor, thus we do not want to relase it.
    ral_tf_ctx->getImpl()->tensor_map[ptr] =
        std::make_pair(*persistent_tensor, 2);
  }
}

template <typename T, int N>
::tao::ral::MemRefType<T, N> ral_tf_bitcast(
    ExecutionContext* ctx, stream_t, ::tao::ral::MemRefType<T, N> input) {
  TAO_VLOG(1) << "ral_tf_bitcast for " << N << "d\n";
  ::tao::ral::MemRefType<T, N> memref = input;

  if (memref.data) {
    ral_tf_bitcast_update_ref_count(ctx, memref.data);
  }

  if (TAO_VLOG_IS_ON(1)) {
    ::tao::ral::print_memref(memref, "ral_tf_bitcast");
  }
  return memref;
}

template <typename T>
::tao::ral::MemRefType<T, 0> ral_tf_bitcast_0d(
    ExecutionContext* ctx, stream_t, ::tao::ral::MemRefType<T, 0> input) {
  TAO_VLOG(1) << "ral_tf_bitcast for 0d";
  ::tao::ral::MemRefType<T, 0> memref = input;

  if (memref.data) {
    ral_tf_bitcast_update_ref_count(ctx, memref.data);
  }

  if (TAO_VLOG_IS_ON(1)) {
    ::tao::ral::print_memref_0d<T>(memref, "ral_tf_bitcast");
  }
  return memref;
}

template <typename T, int N, int M, typename P = int64_t>
::tao::ral::MemRefType<T, M> ral_tf_bitcast(
    ExecutionContext* ctx, stream_t, ::tao::ral::MemRefType<T, N> input,
    ::tao::ral::MemRefType<P, 1> shape) {
  TAO_VLOG(1) << "ral_tf_gpu_d_bitcast for " << M << "d\n";
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
    ral_tf_bitcast_update_ref_count(ctx, memref.data);
  }

  if (TAO_VLOG_IS_ON(1)) {
    ::tao::ral::print_memref(memref, "ral_tf_gpu_d_bitcast");
  }
  return memref;
}

template <typename T, int N, int M, typename P = int64_t>
::tao::ral::MemRefType<T, 0> ral_tf_bitcast_0d(
    ExecutionContext* ctx, stream_t, ::tao::ral::MemRefType<T, N> input,
    ::tao::ral::MemRefType<P, 1> shape) {
  TAO_VLOG(1) << "ral_tf_gpu_d_bitcast_0d for " << M << "d\n";
  ::tao::ral::MemRefType<T, 0> memref;

  memref.basePtr = input.basePtr;
  memref.data = input.data;
  memref.offset = 0;

  if (memref.data) {
    ral_tf_bitcast_update_ref_count(ctx, memref.data);
  }

  if (TAO_VLOG_IS_ON(1)) {
    ::tao::ral::print_memref_0d(memref, "ral_tf_gpu_d_bitcast");
  }
  return memref;
}

void ral_tf_send_output_impl(ExecutionContext* ctx, int64_t output_idx,
                             TensorShape& tensor_shape, void* data,
                             int64_t bytes) {
  if (TAO_VLOG_IS_ON(1)) {
    TAO_VLOG(1) << "setting output #" << output_idx
                << " with shape (rank := " << tensor_shape.dims() << ")";
    for (int i = 0; i < tensor_shape.dims(); ++i) {
      TAO_VLOG(1) << "\tdim #" << i << ": " << tensor_shape.dim_size(i);
    }
  }

  auto ral_tf_ctx = dynamic_cast<RalTfExecutionContext*>(ctx);
  if (!bytes) {
    // fast path for empty tensor
    Tensor* out = nullptr;
    ral_tf_ctx->getOpContext()->allocate_output(output_idx, tensor_shape, &out);
    return;
  }

  Tensor out_tensor;
  Tensor* allocated_tensor = nullptr;
  auto* state = ctx->getResource<RalTFContextState>(kRalTFContextState);
  auto it = ral_tf_ctx->getImpl()->tensor_map.find(data);
  if (it != ral_tf_ctx->getImpl()->tensor_map.end()) {
    allocated_tensor = &it->second.first;
  } else {
    std::lock_guard<std::mutex> lock(state->mu);
    auto it = state->tensor_map.find(data);
    if (it != state->tensor_map.end()) {
      allocated_tensor = &it->second;
    }
  }

  Tensor cpu_tensor;
  if (!allocated_tensor) {
    // cpu buffer
    AllocatorAttributes alloc_attr;
    alloc_attr.set_on_host(true);
    ral_tf_ctx->getOpContext()->allocate_temp(
        DT_UINT8, {static_cast<int64_t>(bytes)}, &cpu_tensor, alloc_attr);
    auto ptr = (buffer_t)cpu_tensor.tensor_data().data();
    std::memcpy(ptr, data, bytes);
    TAO_VLOG(1) << "host memory copy: " << bytes << "@" << data;
    ral_tf_ctx->getImpl()->to_free_host_output_buffers.insert(data);

    allocated_tensor = &cpu_tensor;
  }

#ifdef TF_1_12
  out_tensor.UnsafeCopyFromInternal(
      *allocated_tensor,
      ral_tf_ctx->getOpContext()->expected_output_dtype(output_idx),
      tensor_shape);
#else
  auto s = out_tensor.BitcastFrom(
      *allocated_tensor,
      ral_tf_ctx->getOpContext()->expected_output_dtype(output_idx),
      tensor_shape);
  if (!s.ok()) {
    TAO_VLOG(0) << "bitcast failed: " << s.error_message();
  }
#endif
  TAO_VLOG(1) << "set output #" << output_idx << "@"
              << out_tensor.shape().DebugString();
  ral_tf_ctx->getOpContext()->set_output(output_idx, out_tensor);
}

template <typename T, int N>
void ral_tf_send_output(ExecutionContext* ctx, int64_t output_idx,
                        ::tao::ral::MemRefType<T, N> memref) {
  TAO_VLOG(2) << "tao_ral_send_output #" << output_idx << " for " << N << "d\n";
  if (TAO_VLOG_IS_ON(1)) {
    ::tao::ral::print_memref(memref, "output");
  }
  int64_t bytes = sizeof(T);
  TensorShape tensor_shape;
  for (int i = 0; i < N; ++i) {
    bytes *= memref.sizes[i];
    tensor_shape.AddDim(memref.sizes[i]);
  }
  ral_tf_send_output_impl(ctx, output_idx, tensor_shape, memref.data, bytes);
}

template <typename T>
void ral_tf_send_output_0d(ExecutionContext* ctx, int64_t output_idx,
                           ::tao::ral::MemRefType<T, 0> memref) {
  TAO_VLOG(2) << "tao_ral_send_output #" << output_idx << " for 0d";
  if (TAO_VLOG_IS_ON(1)) {
    ::tao::ral::print_memref_0d(memref, "output");
  }
  TensorShape tensor_shape;
  int64_t bytes = sizeof(T);
  ral_tf_send_output_impl(ctx, output_idx, tensor_shape, memref.data, bytes);
}

#define RAL_REGISTER_IO_FUNC(T, N)                                          \
  template ::tao::ral::MemRefType<T, N> ral_tf_recv_input<T, N>(            \
      ExecutionContext * ctx, int64_t input_idx);                           \
  template void ral_tf_send_output<T, N>(ExecutionContext * ctx,            \
                                         int64_t output_idx,                \
                                         ::tao::ral::MemRefType<T, N>);     \
  TAO_RAL_API(::tao::ral::kRalRecvInput, "cpu", ral_tf_recv_input<T, N>);   \
  TAO_RAL_API(::tao::ral::kRalSendOutput, "cpu", ral_tf_send_output<T, N>); \
  TAO_RAL_API(::tao::ral::kRalBitcast, "gpu", ral_tf_bitcast<T, N>);        \
  TAO_RAL_API(::tao::ral::kRalBitcast, "gpu", ral_tf_bitcast<T, 0, N>);     \
  TAO_RAL_API(::tao::ral::kRalBitcast, "gpu", ral_tf_bitcast<T, 1, N>);     \
  TAO_RAL_API(::tao::ral::kRalBitcast, "gpu", ral_tf_bitcast<T, 2, N>);     \
  TAO_RAL_API(::tao::ral::kRalBitcast, "gpu", ral_tf_bitcast<T, 3, N>);     \
  TAO_RAL_API(::tao::ral::kRalBitcast, "gpu", ral_tf_bitcast<T, 4, N>);     \
  TAO_RAL_API(::tao::ral::kRalBitcast, "gpu", ral_tf_bitcast<T, 5, N>);     \
  TAO_RAL_API(::tao::ral::kRalBitcast, "gpu", ral_tf_bitcast<T, 6, N>);     \
  TAO_RAL_API(::tao::ral::kRalBitcast, "gpu", ral_tf_bitcast<T, 7, N>);     \
  TAO_RAL_API(::tao::ral::kRalBitcast, "gpu", ral_tf_bitcast<T, 8, N>);     \
  TAO_RAL_API(::tao::ral::kRalBitcast, "cpu", ral_tf_bitcast<T, N>);        \
  TAO_RAL_API(::tao::ral::kRalBitcast, "cpu", ral_tf_bitcast<T, 0, N>);     \
  TAO_RAL_API(::tao::ral::kRalBitcast, "cpu", ral_tf_bitcast<T, 1, N>);     \
  TAO_RAL_API(::tao::ral::kRalBitcast, "cpu", ral_tf_bitcast<T, 2, N>);     \
  TAO_RAL_API(::tao::ral::kRalBitcast, "cpu", ral_tf_bitcast<T, 3, N>);     \
  TAO_RAL_API(::tao::ral::kRalBitcast, "cpu", ral_tf_bitcast<T, 4, N>);     \
  TAO_RAL_API(::tao::ral::kRalBitcast, "cpu", ral_tf_bitcast<T, 5, N>);     \
  TAO_RAL_API(::tao::ral::kRalBitcast, "cpu", ral_tf_bitcast<T, 6, N>);     \
  TAO_RAL_API(::tao::ral::kRalBitcast, "cpu", ral_tf_bitcast<T, 7, N>);     \
  TAO_RAL_API(::tao::ral::kRalBitcast, "cpu", ral_tf_bitcast<T, 8, N>);

#define RAL_REGISTER_IO_FUNC_0D(T)                                         \
  template ::tao::ral::MemRefType<T, 0> ral_tf_recv_input_0d<T>(           \
      ExecutionContext * ctx, int64_t input_idx);                          \
  template void ral_tf_send_output_0d<T>(ExecutionContext * ctx,           \
                                         int64_t output_idx,               \
                                         ::tao::ral::MemRefType<T, 0>);    \
  TAO_RAL_API(::tao::ral::kRalRecvInput, "cpu", ral_tf_recv_input_0d<T>);  \
  TAO_RAL_API(::tao::ral::kRalSendOutput, "cpu", ral_tf_send_output_0d<T>) \
  TAO_RAL_API(::tao::ral::kRalBitcast, "gpu", ral_tf_bitcast_0d<T>);       \
  TAO_RAL_API(::tao::ral::kRalBitcast, "gpu", ral_tf_bitcast_0d<T, 0, 0>)  \
  TAO_RAL_API(::tao::ral::kRalBitcast, "gpu", ral_tf_bitcast_0d<T, 1, 0>)  \
  TAO_RAL_API(::tao::ral::kRalBitcast, "gpu", ral_tf_bitcast_0d<T, 2, 0>)  \
  TAO_RAL_API(::tao::ral::kRalBitcast, "gpu", ral_tf_bitcast_0d<T, 3, 0>)  \
  TAO_RAL_API(::tao::ral::kRalBitcast, "gpu", ral_tf_bitcast_0d<T, 4, 0>)  \
  TAO_RAL_API(::tao::ral::kRalBitcast, "gpu", ral_tf_bitcast_0d<T, 5, 0>)  \
  TAO_RAL_API(::tao::ral::kRalBitcast, "gpu", ral_tf_bitcast_0d<T, 6, 0>)  \
  TAO_RAL_API(::tao::ral::kRalBitcast, "gpu", ral_tf_bitcast_0d<T, 7, 0>)  \
  TAO_RAL_API(::tao::ral::kRalBitcast, "gpu", ral_tf_bitcast_0d<T, 8, 0>); \
  TAO_RAL_API(::tao::ral::kRalBitcast, "cpu", ral_tf_bitcast_0d<T, 0, 0>)  \
  TAO_RAL_API(::tao::ral::kRalBitcast, "cpu", ral_tf_bitcast_0d<T, 1, 0>)  \
  TAO_RAL_API(::tao::ral::kRalBitcast, "cpu", ral_tf_bitcast_0d<T, 2, 0>)  \
  TAO_RAL_API(::tao::ral::kRalBitcast, "cpu", ral_tf_bitcast_0d<T, 3, 0>)  \
  TAO_RAL_API(::tao::ral::kRalBitcast, "cpu", ral_tf_bitcast_0d<T, 4, 0>)  \
  TAO_RAL_API(::tao::ral::kRalBitcast, "cpu", ral_tf_bitcast_0d<T, 5, 0>)  \
  TAO_RAL_API(::tao::ral::kRalBitcast, "cpu", ral_tf_bitcast_0d<T, 6, 0>)  \
  TAO_RAL_API(::tao::ral::kRalBitcast, "cpu", ral_tf_bitcast_0d<T, 7, 0>)  \
  TAO_RAL_API(::tao::ral::kRalBitcast, "cpu", ral_tf_bitcast_0d<T, 8, 0>);

RAL_REGISTER_IO_FUNC_0D(float);
RAL_REGISTER_IO_FUNC_0D(double);
RAL_REGISTER_IO_FUNC_0D(Eigen::half);
RAL_REGISTER_IO_FUNC_0D(int8_t);
RAL_REGISTER_IO_FUNC_0D(int32_t);
RAL_REGISTER_IO_FUNC_0D(int64_t);
RAL_REGISTER_IO_FUNC_0D(uint8_t);
RAL_REGISTER_IO_FUNC_0D(bool);
RAL_REGISTER_IO_FUNC(float, 1);
RAL_REGISTER_IO_FUNC(float, 2);
RAL_REGISTER_IO_FUNC(float, 3);
RAL_REGISTER_IO_FUNC(float, 4);
RAL_REGISTER_IO_FUNC(float, 5);
RAL_REGISTER_IO_FUNC(float, 6);
RAL_REGISTER_IO_FUNC(float, 7);
RAL_REGISTER_IO_FUNC(float, 8);
RAL_REGISTER_IO_FUNC(double, 1);
RAL_REGISTER_IO_FUNC(double, 2);
RAL_REGISTER_IO_FUNC(double, 3);
RAL_REGISTER_IO_FUNC(double, 4);
RAL_REGISTER_IO_FUNC(double, 5);
RAL_REGISTER_IO_FUNC(double, 6);
RAL_REGISTER_IO_FUNC(double, 7);
RAL_REGISTER_IO_FUNC(double, 8);
RAL_REGISTER_IO_FUNC(Eigen::half, 1);
RAL_REGISTER_IO_FUNC(Eigen::half, 2);
RAL_REGISTER_IO_FUNC(Eigen::half, 3);
RAL_REGISTER_IO_FUNC(Eigen::half, 4);
RAL_REGISTER_IO_FUNC(Eigen::half, 5);
RAL_REGISTER_IO_FUNC(Eigen::half, 6);
RAL_REGISTER_IO_FUNC(Eigen::half, 7);
RAL_REGISTER_IO_FUNC(Eigen::half, 8);
RAL_REGISTER_IO_FUNC(int8_t, 1);
RAL_REGISTER_IO_FUNC(int8_t, 2);
RAL_REGISTER_IO_FUNC(int8_t, 3);
RAL_REGISTER_IO_FUNC(int8_t, 4);
RAL_REGISTER_IO_FUNC(int8_t, 5);
RAL_REGISTER_IO_FUNC(int8_t, 6);
RAL_REGISTER_IO_FUNC(int8_t, 7);
RAL_REGISTER_IO_FUNC(int8_t, 8);
RAL_REGISTER_IO_FUNC(int32_t, 1);
RAL_REGISTER_IO_FUNC(int32_t, 2);
RAL_REGISTER_IO_FUNC(int32_t, 3);
RAL_REGISTER_IO_FUNC(int32_t, 4);
RAL_REGISTER_IO_FUNC(int32_t, 5);
RAL_REGISTER_IO_FUNC(int32_t, 6);
RAL_REGISTER_IO_FUNC(int32_t, 7);
RAL_REGISTER_IO_FUNC(int32_t, 8);
RAL_REGISTER_IO_FUNC(int64_t, 1);
RAL_REGISTER_IO_FUNC(int64_t, 2);
RAL_REGISTER_IO_FUNC(int64_t, 3);
RAL_REGISTER_IO_FUNC(int64_t, 4);
RAL_REGISTER_IO_FUNC(int64_t, 5);
RAL_REGISTER_IO_FUNC(int64_t, 6);
RAL_REGISTER_IO_FUNC(int64_t, 7);
RAL_REGISTER_IO_FUNC(int64_t, 8);
RAL_REGISTER_IO_FUNC(uint8_t, 1);
RAL_REGISTER_IO_FUNC(uint8_t, 2);
RAL_REGISTER_IO_FUNC(uint8_t, 3);
RAL_REGISTER_IO_FUNC(uint8_t, 4);
RAL_REGISTER_IO_FUNC(uint8_t, 5);
RAL_REGISTER_IO_FUNC(uint8_t, 6);
RAL_REGISTER_IO_FUNC(uint8_t, 7);
RAL_REGISTER_IO_FUNC(uint8_t, 8);
RAL_REGISTER_IO_FUNC(bool, 1);
RAL_REGISTER_IO_FUNC(bool, 2);
RAL_REGISTER_IO_FUNC(bool, 3);
RAL_REGISTER_IO_FUNC(bool, 4);
RAL_REGISTER_IO_FUNC(bool, 5);
RAL_REGISTER_IO_FUNC(bool, 6);
RAL_REGISTER_IO_FUNC(bool, 7);
RAL_REGISTER_IO_FUNC(bool, 8);

}  // namespace tensorflow

namespace tao {
namespace ral {

DEFINE_TAO_TYPE_NAME_HELPER(Eigen::half, "f16");

TAO_RAL_API(::tao::ral::cpu::kRalCpuAlloc, "cpu", tensorflow::ral_tf_cpu_alloc);
TAO_RAL_API(::tao::ral::cpu::kRalCpuAllocPersistent, "cpu",
            tensorflow::ral_tf_cpu_alloc_persistent);
TAO_RAL_API(::tao::ral::cpu::kRalCpuDealloc, "cpu",
            tensorflow::ral_tf_cpu_dealloc);
TAO_RAL_API(::tao::ral::gpu::kRalGpuAlloc, "gpu", tensorflow::ral_tf_gpu_alloc);
TAO_RAL_API(::tao::ral::gpu::kRalGpuAllocPersistent, "gpu",
            tensorflow::ral_tf_gpu_alloc_persistent);
TAO_RAL_API(::tao::ral::gpu::kRalGpuDealloc, "gpu",
            tensorflow::ral_tf_gpu_dealloc);
TAO_RAL_API(::tao::ral::gpu::kRalGpuLaunch, "gpu",
            tensorflow::ral_tf_gpu_launch);
TAO_RAL_API(::tao::ral::gpu::kRalGpuGetStream, "gpu",
            tensorflow::ral_tf_gpu_get_stream);
TAO_RAL_API(::tao::ral::gpu::kRalGpuAsCUStream, "gpu",
            tensorflow::ral_tf_gpu_as_cu_stream);
TAO_RAL_API(::tao::ral::gpu::kRalGpuAsSEStream, "gpu",
            tensorflow::ral_tf_gpu_as_se_stream);
TAO_RAL_API(::tao::ral::gpu::kRalGpuH2D, "gpu", tensorflow::ral_tf_gpu_h2d);
TAO_RAL_API(::tao::ral::gpu::kRalGpuD2H, "gpu", tensorflow::ral_tf_gpu_d2h);
TAO_RAL_API(::tao::ral::gpu::kRalGpuD2D, "gpu", tensorflow::ral_tf_gpu_d2d);
TAO_RAL_API(::tao::ral::gpu::kRalGpuSyncOnStream, "gpu",
            tensorflow::ral_tf_gpu_sync_on_stream);
TAO_RAL_API("tao_ral_cpu_alloc", "cpu", tensorflow::ral_tf_cpu_alloc);
TAO_RAL_API("tao_ral_cpu_free", "cpu", tensorflow::ral_tf_cpu_dealloc);

}  // namespace ral
}  // namespace tao

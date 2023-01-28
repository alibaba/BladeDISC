//===- base_context.cc ----------------------===//
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

#include "mlir/xla/ral/context/base/base_context.h"

#include "mlir/xla/ral/context/common_context_impl.h"
#include "mlir/xla/ral/context/context_util.h"
#include "mlir/xla/ral/ral_helper.h"
#include "mlir/xla/ral/ral_logging.h"
#include "third_party/eigen3/Eigen/Core"

namespace tao {
namespace ral {

DEFINE_TAO_TYPE_NAME_HELPER(Eigen::half, "f16");

BaseContext::BaseContext(BaseContextOption& opt) {
  getOrCreateResource(tao::ral::kRalGlobalConstantState, [opt, this]() {
    auto state = new tao::ral::RalGlobalConstantState;

    if (discEnableGlobalConstantStore()) {
      state->process_level_store =
          ConstStoreRegistrar::Instance().getConstStore(opt.metadata_file_path);
      assert(state->process_level_store);
      return state;
    }

    // The metadata file is loaded once. The data will
    // be erased from metadata file once memcpy is done;
    if ((state->metadata =
             MetadataFile::loadFromFile(opt.metadata_file_path))) {
      return state;
    } else {
      delete state;
      return (RalGlobalConstantState*)nullptr;
    }
  });
}

BaseContext::~BaseContext() {}

BaseExecutionContext::BaseExecutionContext(BaseContext* ctx)
    : ExecutionContext(ctx) {
  onExecutionStart();
}

BaseExecutionContext::~BaseExecutionContext() { onExecutionFinish(); }

void BaseExecutionContext::bindInput(int input_idx, buffer_t buffer,
                                     const buffer_shape_t& shape) {
  Tensor tensor;
  tensor.buffer = buffer;
  tensor.shape = shape;

  inputs.insert(std::make_pair(input_idx, tensor));
  input_ptr_set.insert(buffer);
}

void BaseExecutionContext::bindOutput(
    int output_idx, std::unique_ptr<OutputBufferWrapper>* output) {
  auto it = outputs.find(output_idx);
  if (it == outputs.end()) {
    signalError(Context::FAILURE, "output not found");
    return;
  }

  output->reset(
      new BaseOutputBufferWrapper(it->second.buffer, it->second.shape));
  if (!output_ptr_set.insert(it->second.buffer).second) {
    // This buffer is used as output before, thus is already set a deleter.
    return;
  }

  if (input_ptr_set.count(it->second.buffer)) {
    // This buffer is just a forwording of one input buffer, thus no need to set
    // a deleter.
    return;
  }

  setOutputDeleter(*(output->get()));
}

InternalAllocator::InternalAllocator(alloc_t alloc_func, dealloc_t dealloc_func)
    : alloc_func_(alloc_func), dealloc_func_(dealloc_func) {}

InternalAllocator::~InternalAllocator() { releaseAllFreeBuffers(); }

void InternalAllocator::releaseAllFreeBuffers() {
  for (auto& pair : free_buffers_) {
    for (auto& buffer : pair.second) {
      dealloc_func_(buffer);
    }
  }
}

buffer_t InternalAllocator::alloc(size_t bytes) {
  auto& free_buffer_vec = free_buffers_[bytes];
  if (free_buffer_vec.empty()) {
    buffer_t ptr = alloc_func_(bytes);
    allocated_buffers_[ptr] = bytes;
    return ptr;
  }
  buffer_t ptr = free_buffer_vec.back();
  free_buffer_vec.pop_back();
  allocated_buffers_[buffer_t(ptr)] = bytes;
  return ptr;
}

void InternalAllocator::dealloc(buffer_t buffer) {
  auto it = allocated_buffers_.find(buffer);
  assert(it != allocated_buffers_.end());
  free_buffers_[it->second].push_back(buffer);
  allocated_buffers_.erase(it);
}

// ============================================================================
// ========================== basic kernel api impl
// =============================
// ============================================================================

template <typename T, int N>
tao::ral::MemRefType<T, N> ral_base_cuda_recv_input(ExecutionContext* ctx,
                                                    int64_t input_idx) {
  TAO_VLOG(1) << "ral_base_cuda_recv_input for " << N << "d";
  tao::ral::MemRefType<T, N> memref;

  auto exec_ctx = dynamic_cast<BaseExecutionContext*>(ctx);
  auto it = exec_ctx->inputs.find(input_idx);
  if (it == exec_ctx->inputs.end()) {
    ctx->signalError(Context::FAILURE, "invalid input index");
    return memref;
  }
  auto& tensor = it->second;

  memref = assignMemRef<T, N>(tensor.buffer, tensor.shape);

  if (TAO_VLOG_IS_ON(1)) {
    tao::ral::print_memref(memref, "input");
  }

  return memref;
}

template <typename T>
tao::ral::MemRefType<T, 0> ral_base_cuda_recv_input_0d(ExecutionContext* ctx,
                                                       int64_t input_idx) {
  TAO_VLOG(1) << "ral_base_cuda_recv_input for " << 0 << "d";
  tao::ral::MemRefType<T, 0> memref;

  auto exec_ctx = dynamic_cast<BaseExecutionContext*>(ctx);
  auto it = exec_ctx->inputs.find(input_idx);
  if (it == exec_ctx->inputs.end()) {
    ctx->signalError(Context::FAILURE, "invalid input index");
    return memref;
  }
  auto& tensor = it->second;

  memref = assignMemRef_0d<T>(tensor.buffer);

  if (TAO_VLOG_IS_ON(1)) {
    print_memref_0d<T>(memref, "input");
  }

  return memref;
}

template <typename T, int N>
void ral_base_cuda_send_output(ExecutionContext* ctx, int64_t output_idx,
                               tao::ral::MemRefType<T, N> memref) {
  TAO_VLOG(1) << "ral_base_cuda_send_output for " << N << "d";
  if (TAO_VLOG_IS_ON(1)) {
    tao::ral::print_memref(memref, "output");
  }

  auto exec_ctx = dynamic_cast<BaseExecutionContext*>(ctx);

  Tensor tensor;
  for (int i = 0; i < N; ++i) {
    tensor.shape.push_back(memref.sizes[i]);
  }
  for (int i = 0; i < N; ++i) {
    TAO_VLOG(1) << "tensor dim size = " << tensor.shape[i];
  }
  tensor.buffer = memref.data;

  exec_ctx->outputs.insert(std::make_pair(output_idx, tensor));
  ++(exec_ctx->outputSharedOrder[tensor.buffer]);
  TAO_VLOG(1) << "set_output #" << output_idx;
}

template <typename T>
void ral_base_cuda_send_output_0d(ExecutionContext* ctx, int64_t output_idx,
                                  tao::ral::MemRefType<T, 0> memref) {
  TAO_VLOG(1) << "ral_base_cuda_send_output for " << 0 << "d";
  if (TAO_VLOG_IS_ON(1)) {
    print_memref_0d<T>(memref, "output");
  }

  auto exec_ctx = dynamic_cast<BaseExecutionContext*>(ctx);

  Tensor tensor;
  tensor.buffer = memref.data;
  exec_ctx->outputs.insert(std::make_pair(output_idx, tensor));
  TAO_VLOG(1) << "set_output #" << output_idx;
}

#define RAL_REGISTER_IO_FUNC(T, N)                                             \
  template tao::ral::MemRefType<T, N> ral_base_cuda_recv_input<T, N>(          \
      ExecutionContext * ctx, int64_t input_idx);                              \
  template void ral_base_cuda_send_output<T, N>(                               \
      ExecutionContext * ctx, int64_t output_idx, tao::ral::MemRefType<T, N>); \
  TAO_RAL_API(tao::ral::kRalRecvInput, "cpu", ral_base_cuda_recv_input<T, N>); \
  TAO_RAL_API(tao::ral::kRalSendOutput, "cpu", ral_base_cuda_send_output<T, N>);

#define RAL_REGISTER_IO_FUNC_0D(T)                                             \
  template tao::ral::MemRefType<T, 0> ral_base_cuda_recv_input_0d<T>(          \
      ExecutionContext * ctx, int64_t input_idx);                              \
  template void ral_base_cuda_send_output_0d<T>(                               \
      ExecutionContext * ctx, int64_t output_idx, tao::ral::MemRefType<T, 0>); \
  TAO_RAL_API(tao::ral::kRalRecvInput, "cpu", ral_base_cuda_recv_input_0d<T>); \
  TAO_RAL_API(tao::ral::kRalSendOutput, "cpu", ral_base_cuda_send_output_0d<T>);

RAL_REGISTER_IO_FUNC_0D(float);
RAL_REGISTER_IO_FUNC_0D(double);
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
RAL_REGISTER_IO_FUNC_0D(Eigen::half);
RAL_REGISTER_IO_FUNC(Eigen::half, 1);
RAL_REGISTER_IO_FUNC(Eigen::half, 2);
RAL_REGISTER_IO_FUNC(Eigen::half, 3);
RAL_REGISTER_IO_FUNC(Eigen::half, 4);
RAL_REGISTER_IO_FUNC(Eigen::half, 5);
RAL_REGISTER_IO_FUNC(Eigen::half, 6);
RAL_REGISTER_IO_FUNC(Eigen::half, 7);
RAL_REGISTER_IO_FUNC(Eigen::half, 8);
}  // namespace ral
}  // namespace tao

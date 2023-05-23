//===- base_context.h ----------------------===//
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

#ifndef RAL_CONTEXT_BASE_BASE_CONTEXT_H_
#define RAL_CONTEXT_BASE_BASE_CONTEXT_H_

#include <map>
#include <unordered_map>
#include <unordered_set>

#include "mlir/ral/ral_base.h"
#include "mlir/ral/ral_context.h"

namespace tao {
namespace ral {

struct BaseContextOption {
  std::string metadata_file_path;
  bool cache_workspace_mem_across_execution = false;
};

class BaseContext : public tao::ral::Context {
 public:
  BaseContext(BaseContextOption& opt);
  ~BaseContext();
};

class BaseOutputBufferWrapper : public OutputBufferWrapper {
 public:
  BaseOutputBufferWrapper(buffer_t data, buffer_shape_t shape)
      : data_(data), shape_(shape) {}
  ~BaseOutputBufferWrapper() {
    if (deleter_) deleter_(data_);
  }

  using Deleter = std::function<void(buffer_t)>;

  const_buffer_t data() override { return data_; }
  const buffer_shape_t& shape() override { return shape_; }
  void set_deleter(Deleter deleter) { deleter_ = deleter; }

  // Returns true if this wrapper is the exclusive owner
  bool owned() const override { return owned_; };

  // mark that this wrapper exclusively owns the underlying buffer.
  void markOwned() override { owned_ = true; }

  // Release the ownership of the wrapper buffer.
  // This requires that the buffer is owned by this wrapper.
  void release() override {
    deleter_ = nullptr;
    owned_ = false;
  }

 private:
  buffer_t data_;
  buffer_shape_t shape_;
  Deleter deleter_;
  bool owned_ = false;
};

class InternalAllocator : public Allocator {
 public:
  InternalAllocator(alloc_t alloc_func, dealloc_t dealloc_func);
  ~InternalAllocator();

  void releaseAllFreeBuffers();
  buffer_t alloc(size_t bytes);
  void dealloc(buffer_t buffer);

 private:
  alloc_t alloc_func_;
  dealloc_t dealloc_func_;
  std::map<size_t, std::vector<buffer_t>> free_buffers_;
  std::map<buffer_t, size_t> allocated_buffers_;
};

struct Tensor {
  void* buffer;
  std::vector<int64_t> shape;
};

struct BaseExecutionContext : public tao::ral::ExecutionContext {
  BaseExecutionContext(BaseContext* ctx);
  ~BaseExecutionContext();
  // Sends and Receives inputs/outputs from environment.
  // Input/Output buffer may or may not on device. It's the responsibility
  // of client to make sure each input/output buffer meets the compiler
  // requirement.
  void bindInput(int input_idx, buffer_t buffer,
                 const buffer_shape_t& shape) override;
  void bindOutput(int output_idx,
                  std::unique_ptr<OutputBufferWrapper>* output) override;

  // Record each input ptr. These ptrs will be used to tell if one output is
  // just i/o forwarding.
  std::unordered_set<const_buffer_t> input_ptr_set;
  // In case a same buffer is referred by multiple outputs.
  std::unordered_set<const_buffer_t> output_ptr_set;
  // Input bindings
  std::unordered_map<int32_t, Tensor> inputs;
  // Output bindings
  std::unordered_map<int32_t, Tensor> outputs;
  // Record one underlying output buffer is shared by how many different
  // outputs.
  std::unordered_map<const_buffer_t, int32_t> outputSharedOrder;

 protected:
  virtual void setOutputDeleter(OutputBufferWrapper& output) = 0;
};
}  // namespace ral
}  // namespace tao

#endif  // RAL_CONTEXT_BASE_BASE_CONTEXT_H_
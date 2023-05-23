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

#ifndef TAO_TAO_BRIDGE_EXECUTABLE_H_
#define TAO_TAO_BRIDGE_EXECUTABLE_H_

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "mlir/ral/context/tensorflow/tf_context_impl.h"
#include "tao_bridge/common.h"
#include "tao_bridge/tao_compilation_result.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/stream_executor/device_memory.h"

namespace tensorflow {

namespace tao {

using TaoCompilerResult = CompilationResultProto;
using PrimitiveType = PrimitiveTypeProto;

struct BufferAllocation {
  struct Slice {
    int64 id = -1;
    int64 offset = 0;
    int64 size = 0;
  };

  using Index = int64;

  void* ptr = nullptr;
  int64 size = 0;
};

using BufferSlice = BufferAllocation::Slice;

struct OutputDescription {
  enum OutputType {
    kDefault = 0,
    kConstant = 1,
    kResource = 2,
  };

  OutputType type = kDefault;

  bool is_constant() { return type == kConstant; }
  bool is_resource() { return type == kResource; }
  bool is_default() { return type == kDefault; }

  tensorflow::TensorShape shape;
  tensorflow::Tensor constant_value;
  int64 input_index = -1;
  BufferSlice slice;
};

struct ResourceUpdate {
  DataType dtype;
  tensorflow::TensorShape shape;
  // The input_index of the CompilationResult,
  // which is the associated resource handle
  int64 input_index = -1;
  // the slice of the updated tensor
  BufferSlice slice;
};

struct ConstantDescription {
  std::string constant_global_name;
  std::vector<uint8> constant_value;
  bool constant_emitted_in_ir;
};

// Manage all buffers used during computation
struct BufferAllocations {
  using Index = BufferAllocation::Index;
  Index num_allocations() const { return allocations.size(); }
  Index num_parameters() const {
    return allocation_id_to_parameter_index.size();
  }

  bool is_entry_computation_parameter(Index id) {
    auto iter = allocation_id_to_parameter_index.find(id);
    return iter != allocation_id_to_parameter_index.end();
  }

  bool is_maybe_live_out(Index id) {
    auto iter = maybe_live_out_ids.find(id);
    return iter != maybe_live_out_ids.end();
  }

  bool is_constant(Index id) {
    auto iter = allocation_id_to_constants.find(id);
    return iter != allocation_id_to_constants.end();
  }

  bool is_tuple(Index id) {
    auto iter = tuple_ids.find(id);
    return iter != tuple_ids.end();
  }

  bool is_temp_buffer(Index id) {
    return temp_buffer_id >= 0 && id == temp_buffer_id;
  }

  bool is_thread_local(Index id) {
    auto iter = thread_local_ids.find(id);
    return iter != thread_local_ids.end();
  }

  int parameter_number(Index id) {
    return allocation_id_to_parameter_index.at(id);
  }

  void set_allocation(Index id, void* ptr, int64 size) {
    allocations.at(id).ptr = ptr;
    allocations.at(id).size = size;
  }

  se::DeviceMemoryBase GetDeviceAddress(const BufferSlice& slice) const {
    return se::DeviceMemoryBase(
        static_cast<char*>(allocations.at(slice.id).ptr) + slice.offset,
        slice.size);
  }

  void clear_buffers() {
    for (auto& buffer : allocations) {
      buffer.ptr = nullptr;
    }
  }

  const BufferAllocation& allocation(Index id) const {
    return allocations.at(id);
  }

  Index temp_buffer_id = -1;
  std::vector<BufferAllocation> allocations;
  std::unordered_map<Index, Index> allocation_id_to_parameter_index;
  std::unordered_set<Index> maybe_live_out_ids;
  std::unordered_set<Index> thread_local_ids;
  std::unordered_set<Index> tuple_ids;
  std::unordered_map<Index, ConstantDescription> allocation_id_to_constants;
  std::vector<OutputDescription> output_descriptions;
  std::vector<ResourceUpdate> resource_updates;
};

struct ProfileState {
  int64 execution_time_in_us = -1;

  bool is_valid() { return execution_time_in_us > 0; }
};

class ExecutableRunOptions {
 public:
  ExecutableRunOptions& set_ctx(OpKernelContext* ctx) {
    ctx_ = ctx;
    return *this;
  }
  OpKernelContext* ctx() const { return ctx_; }

  ExecutableRunOptions& set_num_constant_args(int num) {
    num_constant_args_ = num;
    return *this;
  }
  int num_constant_args() const { return num_constant_args_; }

  ExecutableRunOptions& set_variables(std::map<int, OptionalTensor> variables) {
    variables_ = std::move(variables);
    return *this;
  }
  const std::map<int, OptionalTensor>& variables() const { return variables_; }

  ExecutableRunOptions& set_profile_state(ProfileState* state) {
    profile_state_ = state;
    return *this;
  }
  ProfileState* profile_state() const { return profile_state_; }

 private:
  OpKernelContext* ctx_ = nullptr;
  int num_constant_args_ = 0;
  std::map<int, OptionalTensor> variables_;
  ProfileState* profile_state_ = nullptr;
};

class Executable {
 public:
  Executable(const std::string& compiled_result_file);
  virtual ~Executable();

  virtual Status Run(const ExecutableRunOptions& options);

  // 1, Load TaoCompilerResult from file
  // 2, Parse TaoCompilerResult and prepare to run
  virtual Status Init();

  std::string compiled_result_file() const { return compiled_result_file_; }

  virtual void DumpToFile(const std::string& filename) const;

  virtual std::string target_device() const = 0;

 protected:
  virtual Status InitBufferAllocations(TaoCompilerResult*);
  virtual Status InitImpl(const TaoCompilerResult*) = 0;
  // allocate buffers and binding names
  virtual Status PreRunProcess(const ExecutableRunOptions& options,
                               BufferAllocations& allocations,
                               std::vector<Tensor>& output_tensors);
  // Do real computation here
  virtual Status RunImpl(const ExecutableRunOptions& options,
                         BufferAllocations& allocations) = 0;
  // populate outputs and maybe free temp buffers
  virtual Status PostRunProcess(const ExecutableRunOptions& options,
                                BufferAllocations& allocations,
                                std::vector<Tensor>& output_tensors);

  virtual Status StartProfiler(const ExecutableRunOptions& options) {
    return Status::OK();
  }
  virtual Status StopProfiler(const ExecutableRunOptions& options) {
    return Status::OK();
  }

  const TaoCompilerResult& tao_compiler_result() const {
    CHECK(compiled_result_.get() != nullptr);
    return *compiled_result_;
  }

  TaoCompilerResult& tao_compiler_result() {
    CHECK(compiled_result_.get() != nullptr);
    return *compiled_result_;
  }

  RalTfContext* ral_context() { return ral_ctx_.get(); }

  std::unique_ptr<RalTfContext>& mutable_ral_context() { return ral_ctx_; }

  BufferAllocations& buffers() { return buffers_; }
  const BufferAllocations& buffers() const { return buffers_; }

  std::vector<Tensor>& output_tensors() { return output_tensors_; }
  std::unique_ptr<TaoCompilerResult> compiled_result_;

 private:
  std::string compiled_result_file_;

  BufferAllocations buffers_;

  std::vector<Tensor> output_tensors_;

  std::unique_ptr<RalTfContext> ral_ctx_;

  TF_DISALLOW_COPY_AND_ASSIGN(Executable);
};

class ExecutableFactory {
 public:
  using ExecutableConstructor =
      std::function<std::unique_ptr<Executable>(const std::string&)>;
  std::unique_ptr<Executable> NewExecutable(const std::string& device_type,
                                            const std::string& proto_file);
  static ExecutableFactory& Global();
  bool RegisterExecutable(const std::string& device_type,
                          ExecutableConstructor);

 private:
  ExecutableFactory() = default;
  std::unordered_map<std::string, ExecutableConstructor> constructors_;
};

struct Executableregistrar {
  using ExecutableConstructor = ExecutableFactory::ExecutableConstructor;
  Executableregistrar(const std::string& device_type,
                      ExecutableConstructor ctor) {
    ExecutableFactory::Global().RegisterExecutable(device_type, ctor);
  }
};

#define TAO_REGISTER_EXECUTABLE(device, ctor) \
  TAO_REGISTER_EXECUTABLE_IMPL(device, ctor, __COUNTER__)

#define TAO_REGISTER_EXECUTABLE_IMPL(device, ctor, ctr)         \
  Executableregistrar INTERNAL_REGISTER_TAO_EXECUBALE_NAME(ctr) \
      TF_ATTRIBUTE_UNUSED(device, ctor)

// __COUNTER__ must go through another macro to be properly expanded
#define INTERNAL_REGISTER_TAO_EXECUBALE_NAME(ctr) ___##ctr##__object_

}  // namespace tao

}  // namespace tensorflow

#endif  // TAO_TAO_BRIDGE_EXECUTABLE_H_

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

#include "tao_bridge/executable.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace tensorflow {

namespace tao {

Status Executable::Init() {
  // Load TaoCompilerResult from proto file
  compiled_result_.reset(new TaoCompilerResult);
  // TF_RETURN_IF_ERROR(ReadBinaryProto(tensorflow::Env::Default(),
  //     compiled_result_file(), compiled_result_.get()));
  TF_RETURN_IF_ERROR(ReadTextProto(tensorflow::Env::Default(),
                                   compiled_result_file(),
                                   compiled_result_.get()));

  // Init BufferAllocations
  TF_RETURN_IF_ERROR(InitBufferAllocations(compiled_result_.get()));

  // Init device-specific fields
  TF_RETURN_IF_ERROR(InitImpl(compiled_result_.get()));

  VLOG(2) << "Finish Init Executable From " << compiled_result_file();
  return Status::OK();
}

Status Executable::InitBufferAllocations(TaoCompilerResult* result) {
  TF_RET_CHECK(result != nullptr);
  auto& allocations = buffers().allocations;
  allocations.resize(result->buffer_allocation_size());
  VLOG(1) << "allocations size = " << result->buffer_allocation_size();
  for (int i = 0; i < result->buffer_allocation_size(); ++i) {
    allocations[i].size = result->buffer_allocation(i).size();
    VLOG(1) << "size = " << allocations[i].size;
    if (result->buffer_allocation(i).is_parameter()) {
      int parameter_index = result->buffer_allocation(i).parameter_index();
      auto iter = buffers().allocation_id_to_parameter_index.insert(
          std::make_pair(i, parameter_index));
      VLOG(1) << "InitBufferAllocations: bind Allocation #" << i << " to "
              << "paraemter #" << parameter_index;
      TF_RET_CHECK(iter.second);
    } else if (result->buffer_allocation(i).is_maybe_live_out()) {
      auto iter = buffers().maybe_live_out_ids.insert(i);
      VLOG(1) << "InitBufferAllocations: bind Allocation #" << i << " to "
              << "maybe_live_out";
      TF_RET_CHECK(iter.second);
    } else if (result->buffer_allocation(i).is_thread_local()) {
      auto iter = buffers().thread_local_ids.insert(i);
      VLOG(1) << "InitBufferAllocations: bind Allocation #" << i << " to "
              << "thread_local";
      TF_RET_CHECK(iter.second);
    } else if (result->buffer_allocation(i).is_temp_buffer()) {
      // There should be only one temp buffer for gpu.
      TF_RET_CHECK(buffers().temp_buffer_id < 0);
      buffers().temp_buffer_id = i;
      VLOG(1) << "InitBufferAllocations: bind Allocation #" << i << " to "
              << "temp_buffer";
    } else if (result->buffer_allocation(i).is_constant()) {
      ConstantDescription constant_dscrp;
      constant_dscrp.constant_emitted_in_ir =
          result->buffer_allocation(i).constant_emitted_in_ir();
      constant_dscrp.constant_global_name =
          result->buffer_allocation(i).constant_global_name();
      auto& constant = result->buffer_allocation(i).constant_tensor();
      constant_dscrp.constant_value = std::vector<uint8>(
          constant.data(), constant.data() + constant.size());

      auto iter = buffers().allocation_id_to_constants.insert(
          std::make_pair(i, constant_dscrp));
      VLOG(1) << "InitBufferAllocations: bind Allocation #" << i << " to "
              << "constant (emitted in ir = "
              << constant_dscrp.constant_emitted_in_ir << ")";
      TF_RET_CHECK(iter.second);
    } else if (result->buffer_allocation(i).is_tuple()) {
      auto iter = buffers().tuple_ids.insert(i);
      VLOG(1) << "InitBufferAllocations: bind Allocation #" << i << " to "
              << "tuple";
      TF_RET_CHECK(iter.second);
    } else {
      return errors::Internal("Not supported BufferAllocation type");
    }
  }

  auto& outputs = buffers().output_descriptions;
  outputs.resize(result->outputs_size());
  VLOG(2) << "output_size: " << result->outputs_size();
  for (int i = 0; i < result->outputs_size(); ++i) {
    auto& output = result->outputs(i);
    if (output.output_type() == OutputDescriptionProto_OutputType_DEFAULT) {
      outputs[i].type = OutputDescription::OutputType::kDefault;
      std::vector<int64> dims;
      for (int j = 0; j < output.shape().dims_size(); ++j) {
        dims.push_back(output.shape().dims(j));
      }
      outputs[i].shape = tensorflow::TensorShape(dims);
      auto id = output.slice().buffer_allocation_index();
      TF_RET_CHECK(id >= 0 && id < buffers().num_allocations());
      outputs[i].slice.id = id;
      TF_RET_CHECK(output.slice().offset() == 0);
      TF_RET_CHECK(output.slice().size() <= allocations[id].size);
    } else if (output.output_type() ==
               OutputDescriptionProto_OutputType_CONSTANT) {
      outputs[i].type = OutputDescription::OutputType::kConstant;
      VLOG(2) << "constant output" << i;
      std::vector<int64> dims;
      for (int j = 0; j < output.shape().dims_size(); ++j) {
        dims.push_back(output.shape().dims(j));
      }
      outputs[i].shape = tensorflow::TensorShape(dims);
      if (outputs[i].shape.num_elements() != 0) {
        tensorflow::TensorProto tensor_proto;
        tensor_proto.ParseFromString(output.constant_value());
        TF_RET_CHECK(outputs[i].constant_value.FromProto(tensor_proto));
      }
    } else {
      TF_RET_CHECK(output.output_type() ==
                   OutputDescriptionProto_OutputType_RESOURCE);
      outputs[i].type = OutputDescription::OutputType::kResource;
      outputs[i].input_index = output.input_index();
      TF_RET_CHECK(outputs[i].input_index >= 0);
      std::vector<int64> dims;
      for (int j = 0; j < output.shape().dims_size(); ++j) {
        dims.push_back(output.shape().dims(j));
      }
      outputs[i].shape = tensorflow::TensorShape(dims);
    }
  }

  auto& updates = buffers().resource_updates;
  updates.resize(result->resource_updates_size());
  VLOG(2) << "resource_update_size: " << result->resource_updates_size();
  for (int i = 0; i < result->resource_updates_size(); ++i) {
    auto& update = result->resource_updates(i);
    updates[i].input_index = update.input_index();
    TF_RET_CHECK(updates[i].input_index >= 0);
    std::vector<int64> dims;
    for (int j = 0; j < update.shape().dims_size(); ++j) {
      dims.push_back(update.shape().dims(j));
    }
    updates[i].shape = tensorflow::TensorShape(dims);
    auto id = update.slice().buffer_allocation_index();
    TF_RET_CHECK(id >= 0 && id < buffers().num_allocations());
    updates[i].slice.id = id;
    TF_RET_CHECK(update.slice().offset() == 0);
    TF_RET_CHECK(update.slice().size() <= allocations[id].size);
    // TODO: check tao::DataType is equal to tensorflow::Datatype
    updates[i].dtype =
        static_cast<tensorflow::DataType>(static_cast<int32>(update.dtype()));
  }

  return Status::OK();
}

Executable::Executable(const string& compiled_result_file)
    : compiled_result_file_(compiled_result_file) {}

Executable::~Executable() { VLOG(2) << "Executable::~Executable() is called"; }

Status Executable::Run(const ExecutableRunOptions& options) {
  auto allocations_copy = buffers_;
  auto output_tensors_copy = output_tensors_;
  TF_RETURN_IF_ERROR(StartProfiler(options));
  TF_RETURN_IF_ERROR(
      PreRunProcess(options, allocations_copy, output_tensors_copy));
  TF_RETURN_IF_ERROR(RunImpl(options, allocations_copy));
  TF_RETURN_IF_ERROR(
      PostRunProcess(options, allocations_copy, output_tensors_copy));
  TF_RETURN_IF_ERROR(StopProfiler(options));
  return Status::OK();
}

Status Executable::PreRunProcess(const ExecutableRunOptions& options,
                                 BufferAllocations& allocations,
                                 std::vector<Tensor>& output_tensors) {
  auto ctx = options.ctx();
  int num_const_args = options.num_constant_args();

  // Binding Parameters
  for (auto& pair : allocations.allocation_id_to_parameter_index) {
    auto& idx = pair.first;
    auto& arg_num = pair.second;
    TF_RET_CHECK(allocations.is_entry_computation_parameter(idx));
    const Tensor* t = nullptr;
    TF_RET_CHECK(arg_num >= num_const_args);
    if (options.variables().count(arg_num)) {
      t = &(options.variables().at(arg_num).value);
    } else {
      t = &(ctx->input(arg_num));
    }
    allocations.set_allocation(idx, const_cast<char*>(t->tensor_data().data()),
                               t->tensor_data().size());
    VLOG(2) << "Binding BufferAllocation #" << idx << " to parameter #"
            << arg_num;
  }

  // Tensorflow::Tensor does not support construction from an
  // allocated buffer (corresponding construction function is
  // private). Thus we need to allocate and register outputs
  // buffer here in order to avoid memory copy.
  int output_num = 0;
  output_tensors.clear();
  auto& outputs = allocations.output_descriptions;
  auto& updates = allocations.resource_updates;
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (outputs[i].is_default()) {
      output_tensors.emplace_back();
      Tensor* tensor = nullptr;
      TF_RETURN_IF_ERROR(ctx->allocate_output(i, outputs[i].shape, &tensor));
      TF_RET_CHECK(tensor != nullptr);
      allocations.set_allocation(
          outputs[i].slice.id, const_cast<char*>(tensor->tensor_data().data()),
          tensor->tensor_data().size());
      ++output_num;
      VLOG(2) << "PreAllocate Output #" << i
              << ", addr: " << allocations.allocation(outputs[i].slice.id).ptr
              << ", size: " << allocations.allocation(outputs[i].slice.id).size;
    }
  }

  for (size_t i = 0; i < updates.size(); ++i) {
    output_tensors.emplace_back();
    auto& tensor = output_tensors.back();
    // TODO: check if it is safe to use allocate_temp
    // for resource update.
    ctx->allocate_temp(updates[i].dtype, updates[i].shape, &tensor);
    allocations.set_allocation(updates[i].slice.id,
                               const_cast<char*>(tensor.tensor_data().data()),
                               tensor.tensor_data().size());
    ++output_num;
    VLOG(2) << "PreAllocate ResourceUpdate #" << i;
  }

  VLOG(2) << "Finish Executable::PreRunProcess...";

  return Status::OK();
}

Status Executable::PostRunProcess(const ExecutableRunOptions& options,
                                  BufferAllocations& allocations,
                                  std::vector<Tensor>& output_tensors) {
  auto ctx = options.ctx();
  se::Stream* stream =
      ctx->op_device_context() ? ctx->op_device_context()->stream() : nullptr;
  auto& outputs = allocations.output_descriptions;
  auto& updates = allocations.resource_updates;
  TF_RET_CHECK(outputs.size() == static_cast<size_t>(ctx->num_outputs()));

  int output_num = 0;
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (outputs[i].is_constant()) {
      const Tensor& const_tensor = outputs[i].constant_value;
      Tensor* output_tensor;
      const size_t total_bytes = const_tensor.TotalBytes();
      if (stream && total_bytes > 0) {
        // Copy host -> device. (Empty tensors don't have backing buffers.)
        // Manually allocate memory using an XlaTensorBuffer so we can allocate
        // as much memory as the device requires (as given by
        // GetByteSizeRequirement). This avoids XlaTransferManager having to
        // reallocate the device buffer later.
        VLOG(1) << "Constant output tensor on device";

        TF_RETURN_IF_ERROR(
            ctx->allocate_output(i, const_tensor.shape(), &output_tensor));

        auto to_ptr = [](const Tensor& tensor) {
          return const_cast<void*>(
              static_cast<const void*>(tensor.tensor_data().data()));
        };

        // Copy From Host To Device
        se::DeviceMemoryBase dst_mem(to_ptr(*output_tensor));
        stream->ThenMemcpy(&dst_mem, to_ptr(const_tensor), total_bytes);
      } else {
        // No copy required.
        ctx->set_output(
            i, Tensor(ctx->expected_output_dtype(i), outputs[i].shape));
      }
      VLOG(2) << "set kConstant output #" << i;
    } else if (outputs[i].is_resource()) {
      ctx->set_output(i, ctx->input(outputs[i].input_index));
    } else {
      TF_RET_CHECK(outputs[i].is_default());
      // PreAllocate in PreRunProcess, thus nothing needs to do
      VLOG(2) << "set kDefault output #" << i;
      ++output_num;
    }
  }

  for (size_t i = 0; i < updates.size(); ++i) {
    int input_index = updates[i].input_index;
    VLOG(2) << "update #" << i << " binding to input #" << input_index;
    Var* variable = nullptr;
    // TODO(b/35625933): tensorflow::Var should contain a PersistentTensor,
    // not a Tensor.
    TF_RETURN_IF_ERROR(
        LookupOrCreateResource<Var>(ctx, HandleFromInput(ctx, input_index),
                                    &variable, [&updates, i](Var** ptr) {
                                      *ptr = new Var(updates[i].dtype);
                                      return Status::OK();
                                    }));

    core::ScopedUnref s(variable);
    mutex_lock ml(*variable->mu());
    if (variable->tensor()->dtype() != updates[i].dtype) {
      return errors::Internal("Mismatched type in variable write");
    }
    *variable->tensor() = output_tensors[output_num];
    ++output_num;
  }

  output_tensors.clear();
  allocations.clear_buffers();
  return Status::OK();
}

void Executable::DumpToFile(const std::string& filename) const {
  CHECK(WriteBinaryProto(tensorflow::Env::Default(), filename,
                         *compiled_result_.get())
            .ok());
}

std::unique_ptr<Executable> ExecutableFactory::NewExecutable(
    const std::string& device_type, const std::string& proto_file) {
  auto& instance = Global();
  auto iter = instance.constructors_.find(device_type);
  if (iter != instance.constructors_.end()) {
    return iter->second(proto_file);
  } else {
    return nullptr;
  }
}

ExecutableFactory& ExecutableFactory::Global() {
  static ExecutableFactory instance;
  return instance;
}

bool ExecutableFactory::RegisterExecutable(const std::string& device_type,
                                           ExecutableConstructor ctor) {
  auto& instance = Global();
  return instance.constructors_.insert(std::make_pair(device_type, ctor))
      .second;
}

}  // namespace tao

}  // namespace tensorflow

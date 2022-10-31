// Copyright 2022 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pytorch_blade/compiler/tensorrt/tensorrt_engine_context.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/script.h>

#include "pytorch_blade/compiler/jit/shape_type_spec.h"
#include "pytorch_blade/compiler/tensorrt/bridge/tensorrt_common.h"
#include "pytorch_blade/compiler/tensorrt/bridge/tensorrt_logger.h"

namespace torch {
namespace blade {
namespace tensorrt {
at::ScalarType NvDataType2TorchDataType(nvinfer1::DataType dtype) {
  static std::map<nvinfer1::DataType, at::ScalarType> d2t = {
      {nvinfer1::DataType::kFLOAT, torch::kFloat},
      {nvinfer1::DataType::kHALF, torch::kFloat16},
      {nvinfer1::DataType::kINT8, torch::kInt8},
      {nvinfer1::DataType::kINT32, torch::kInt32},
      {nvinfer1::DataType::kBOOL, torch::kBool}};
  auto it = d2t.find(dtype);
  if (it != d2t.end()) {
    return it->second;
  } else {
    std::stringstream ss;
    ss << "Unsupported nvinfer dtype: " << NvDataType2String(dtype);
    throw std::runtime_error(ss.str());
  }
}

namespace {
ShapeType NvInferType2TorchType(
    const nvinfer1::Dims& dims,
    const nvinfer1::DataType& data_type) {
  ShapeType torch_type;
  torch_type.shape.resize(dims.nbDims);
  for (int j = 0; j < dims.nbDims; ++j) {
    // raise int32_t to int64_t
    torch_type.shape[j] = static_cast<int64_t>(dims.d[j]);
  }

  torch_type.type = NvDataType2TorchDataType(data_type);
  return torch_type;
}
} // namespace

void NvinferExecutionContextDeleter(nvinfer1::IExecutionContext* ctx) {
  if (ctx != nullptr) {
    ctx->destroy();
  }
}

// Check if every tensor in a list of tensors matches the current
// device.
bool TRTContext::CheckCurrentDevice(const at::List<at::Tensor>& inputs) const {
  if (inputs.empty()) {
    return true;
  }

  torch::Device cur_cuda_device =
      torch::Device(torch::kCUDA, c10::cuda::current_device());

  auto& inputs_info = engine_state_->inputs;
  TORCH_CHECK(inputs_info.size() == inputs.size());
  for (size_t k = 0; k < inputs.size(); ++k) {
    at::Tensor inp = inputs[k];
    auto device = inputs_info[k].device;
    if (device == "cuda" && inp.device() != cur_cuda_device) {
      return false;
    }
  }
  return true;
}

std::shared_ptr<nvinfer1::IExecutionContext> TRTContext::GetExecutionContext(
    c10::cuda::CUDAStream& stream,
    const at::List<at::Tensor>& inputs) {
  UpdateProfileIfNeed(inputs);
  auto found = contexts_map_.find(stream);
  if (found != contexts_map_.end() &&
      optimization_profile_ < (found->second).size()) {
    auto contextes = found->second;
    return contextes[optimization_profile_];
  } else {
    LOG(INFO) << "Create a new context for stream: " << stream.id()
              << std::endl;
    nvinfer1::IExecutionContext* ctx = engine_->createExecutionContext();
    if (ctx == nullptr) {
      throw std::runtime_error(
          "Failed to create execution context, it might have been created too many.");
    }
    ctx->setOptimizationProfile(optimization_profile_);
    std::shared_ptr<nvinfer1::IExecutionContext> context(
        ctx, NvinferExecutionContextDeleter);
    contexts_map_[stream].push_back(context);
    return context;
  }
}

TRTContext::TRTContext(std::shared_ptr<State> state)
    : engine_state_(state), tensorrt_device_(c10::cuda::current_device()) {
  TORCH_CHECK(engine_state_ != nullptr, "The engine state is empty");
  const auto& engine_data = engine_state_->engine_bytes;
  auto* engine_ptr = CreateInferRuntime(engine_data);
  TORCH_CHECK(engine_ptr);
  engine_.reset(engine_ptr);

  // settings for multi optimization profiles
  profile_num_ = engine_ptr->getNbOptimizationProfiles();
  input_bind_indices_.resize(profile_num_);
  output_bind_indices_.resize(profile_num_);

  const auto& graph_inputs = engine_state_->inputs;
  const auto& graph_outputs = engine_state_->outputs;
  // The input and output information needs to be set in the order of the
  // profile. Or there will be -1 in the output dims of tensor in previous
  // profiles. Bindings for multiple optimization profiles can be found here:
  // https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#opt_profiles_bindings
  for (int profile_index = 0; profile_index < profile_num_; profile_index++) {
    std::stringstream name_suffix_ss;
    if (profile_index > 0) {
      name_suffix_ss << " [profile " << profile_index << "]";
    }
    std::string name_suffix = name_suffix_ss.str();
    const auto& graph_inputs = engine_state_->inputs;
    auto& input_bind_indices = input_bind_indices_[profile_index];
    auto& output_bind_indices = output_bind_indices_[profile_index];

    input_bind_indices.resize(graph_inputs.size());
    for (int k = 0; k < graph_inputs.size(); ++k) {
      // input_blob_names is in profile order like:
      // [x1, x2, x1 [profile 1], x2 [profile 1], ...]
      const auto& graph_input = graph_inputs[k];
      std::string inp_name = graph_input.name + name_suffix;
      int bind_index = engine_ptr->getBindingIndex(inp_name.c_str());
      input_bind_indices[k] = bind_index;
    }

    output_bind_indices.resize(graph_outputs.size());
    for (int k = 0; k < graph_outputs.size(); ++k) {
      const auto& graph_output = graph_outputs[k];
      std::string out_name = graph_output.name + name_suffix;
      int bind_index = engine_ptr->getBindingIndex(out_name.c_str());
      TORCH_CHECK(!engine_ptr->bindingIsInput(bind_index));
      output_bind_indices[k] = bind_index;
    }
  }
}

std::string TRTContext::SerializeAsString() const {
  if (engine_ != nullptr) {
    auto out = engine_->serialize();
    if (out != nullptr) {
      std::string engine_bytes((char*)out->data(), out->size());
      out->destroy();
      return engine_bytes;
    }
  }
  return "";
}

// Setup binding buffers to the input/output blobs on the GPU.
// Input/output blob mem ptr should be provided by the caller,
// the TRTEngine is not the owner of the blobs' cuda memory.
void TRTContext::BindingInputs(
    const at::List<at::Tensor>& inputs,
    std::vector<void*>& binding_buffers) const {
  for (size_t k = 0; k < inputs.size(); ++k) {
    at::Tensor inp_tensor = inputs[k];
    int bind_index = input_bind_indices_[optimization_profile_][k];
    // The subgraph input should have been eliminated in TensorRT engine
    if (bind_index < 0) {
      continue;
    }
    // assert ctu_tensor is in contiguous layout
    DCHECK(inp_tensor.is_cuda());
    DCHECK(inp_tensor.is_contiguous());
    binding_buffers[bind_index] = inp_tensor.data_ptr();
  }
}

at::List<at::Tensor> TRTContext::CreateAndBindingOutputs(
    std::vector<void*>& binding_buffers,
    std::shared_ptr<nvinfer1::IExecutionContext>& context) const {
  at::List<at::Tensor> outputs{};
  const auto& graph_outputs = engine_state_->outputs;
  outputs.reserve(graph_outputs.size());
  // Note: output_blob_names is aligned to the TRTEngine Operator's outputs, it
  // maybe duplicates. However, duplicate outputs to TensorRT engine are not
  // allowed. So, we use output name mapping between the TRTEngine Operator and
  // TensorRT Engine.
  std::unordered_map<std::string, at::Tensor> dedup_tensors;
  for (size_t k = 0; k < graph_outputs.size(); ++k) {
    const auto& name = graph_outputs[k].name;
    const auto& found = dedup_tensors.find(name);
    auto bind_index = output_bind_indices_[optimization_profile_][k];
    if (found != dedup_tensors.end()) {
      CHECK(found->second.data_ptr() == binding_buffers[bind_index]);
      outputs.push_back(found->second);
      continue;
    }
    const auto& dims = context->getBindingDimensions(bind_index);
    const auto& data_type = engine_->getBindingDataType(bind_index);
    ShapeType torch_type = NvInferType2TorchType(dims, data_type);
    // allocate output tensors
    auto option = torch::device(torch::kCUDA)
                      .dtype(torch_type.type)
                      .memory_format(torch::MemoryFormat::Contiguous);
    at::Tensor out_tensor = torch::empty(torch_type.shape, option);
    binding_buffers[output_bind_indices_[optimization_profile_][k]] =
        out_tensor.data_ptr();
    outputs.push_back(out_tensor);
    dedup_tensors[name] = out_tensor;
  }
  return outputs;
}

bool TRTContext::IsInRange(
    const at::List<at::Tensor>& inputs,
    int64_t profile_index) {
  /* Determine whether the inputs shapes in ranges of the profile. */
  const auto& profile_input_bind_indices = input_bind_indices_[profile_index];

  for (size_t k = 0; k < inputs.size(); ++k) {
    auto const& bind_index = profile_input_bind_indices[k];
    if (bind_index < 0) {
      // skip invalid input
      continue;
    }
    at::Tensor inp_tensor = inputs[k];
    const auto& inp_shape = inp_tensor.sizes();
    // bindingIndex of getProfileDimensions must belong to the given
    // profile, or be between 0 and bindingsPerProfile-1
    const auto& min_dims = engine_->getProfileDimensions(
        bind_index, profile_index, nvinfer1::OptProfileSelector::kMIN);
    const auto& max_dims = engine_->getProfileDimensions(
        bind_index, profile_index, nvinfer1::OptProfileSelector::kMAX);

    TORCH_CHECK(min_dims.nbDims == max_dims.nbDims);
    if (inp_shape.size() != min_dims.nbDims) {
      return false;
    }
    for (int j = 0; j < min_dims.nbDims; j++) {
      if (inp_shape[j] > max_dims.d[j] || inp_shape[j] < min_dims.d[j]) {
        return false;
      }
    }
  }

  return true;
}

bool TRTContext::IsInRange(const at::List<at::Tensor>& inputs) {
  for (int64_t j = 0; j < profile_num_; ++j) {
    // start iterate from optimization_profile_
    auto i = (j + optimization_profile_) % profile_num_;
    if (IsInRange(inputs, i)) {
      return true;
    }
  }
  return false;
}

void TRTContext::UpdateProfileIfNeed(const at::List<at::Tensor>& inputs) {
  /* Update profile according to the inputs' shapes if multiple profiles are
   * used */
  if (profile_num_ <= 1) {
    return;
  }

  for (int64_t j = 0; j < profile_num_; ++j) {
    // start iterate from optimization_profile_
    auto i = (j + optimization_profile_) % profile_num_;
    if (IsInRange(inputs, i)) {
      optimization_profile_ = i;
    }
  }
}

bool TRTContext::ChangingShape(
    const at::List<at::Tensor>& inputs,
    std::shared_ptr<nvinfer1::IExecutionContext>& context) {
  if (!IsInRange(inputs, optimization_profile_)) {
    return false;
  }

  const auto& graph_inputs = engine_state_->inputs;
  for (size_t k = 0; k < inputs.size(); ++k) {
    at::Tensor inp_tensor = inputs[k];
    int bind_index = input_bind_indices_[optimization_profile_][k];
    // The subgraph input should have been eliminated in TensorRT engine
    if (bind_index < 0) {
      continue;
    }
    auto dims = context->getBindingDimensions(bind_index);
    auto inp_size = inp_tensor.sizes();

    TORCH_CHECK(dims.nbDims == inp_size.size());
    for (int i = 0; i < inp_size.size(); i++) {
      dims.d[i] = inp_size[i];
    }
    context->setBindingDimensions(bind_index, dims);
  }
  return true;
}

at::List<at::Tensor> TRTContext::PreProcessInputs(
    const at::List<at::Tensor>& inputs,
    std::shared_ptr<nvinfer1::IExecutionContext>& context) {
  // TODO: we currently only support inputs on the same device as tensorrt
  TORCH_CHECK(tensorrt_device_ == c10::cuda::current_device());
  TORCH_CHECK(CheckCurrentDevice(inputs));
  TORCH_CHECK(ChangingShape(inputs, context));

  const auto& graph_inputs = engine_state_->inputs;
  // pre-process the input bindings
  at::List<at::Tensor> cuda_ctu_inputs;
  for (int k = 0; k < inputs.size(); ++k) {
    // make sure the input is in contiguous layout
    at::Tensor inp_tensor = inputs[k];
    auto dtype = graph_inputs[k].scalar_type;
    // TODO: the option follow not valid for contiguous
    // auto option = torch::device(torch::kCUDA)
    //                   .dtype(torch_type.type)
    //                   .memory_format(torch::MemoryFormat::Contiguous);
    // auto ctu_tensor = inp_tensor.to(option);
    auto option = torch::device(torch::kCUDA).dtype(dtype);
    auto ctu_tensor = inp_tensor.to(option).contiguous();
    // add to cuda_ctu_inputs, make the ctu_tensor has lifetime out of the loop
    cuda_ctu_inputs.push_back(ctu_tensor);
  }

  return cuda_ctu_inputs;
}

at::List<at::Tensor> TRTContext::Execute(const at::List<at::Tensor>& inputs) {
  std::vector<void*> binding_buffers(engine_->getNbBindings());

  // TODO: we assume the inputs are all computed on the current cuda
  // stream so that we get rid of stream synchronization
  c10::cuda::CUDAStream stream =
      c10::cuda::getCurrentCUDAStream(tensorrt_device_);

  // Thread-safe GetExecutionContext and modify contexts_map_.
  // We launch tensorrt engine kernels to the stream with mutex lock
  // synchronously.
  std::lock_guard<std::mutex> guard(lock_);
  // the inputs is pass as args to select a suitable profile
  auto context = GetExecutionContext(stream, inputs);
  at::List<at::Tensor> cuda_ctu_inputs = PreProcessInputs(inputs, context);

  // Input/output CUDA memory bindings
  BindingInputs(cuda_ctu_inputs, binding_buffers);
  at::List<at::Tensor> outputs =
      CreateAndBindingOutputs(binding_buffers, context);
  context->enqueueV2(&binding_buffers[0], stream, nullptr);
  return PostProcessOutputs(outputs);
}

at::List<at::Tensor> TRTContext::PostProcessOutputs(
    const at::List<at::Tensor>& outputs) const {
  at::List<at::Tensor> new_outputs;
  auto const& graph_outputs = engine_state_->outputs;
  for (size_t k = 0; k < outputs.size(); ++k) {
    at::Tensor out = outputs[k];
    at::ScalarType type = graph_outputs[k].scalar_type;
    new_outputs.push_back(out.to(type));
  }
  return new_outputs;
}

} // namespace tensorrt
} // namespace blade
} // namespace torch

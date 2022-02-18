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

#include "compiler/mlir/runtime/ral_context.h"

#include <dlfcn.h>

#include <c10/core/CPUAllocator.h>

#ifdef TORCH_BLADE_BUILD_WITH_CUDA
#ifdef TORCH_BLADE_USE_ROCM
#include <c10/hip/HIPCachingAllocator.h>
#include <c10/hip/HIPFunctions.h>
#else // TORCH_BLADE_USE_ROCM
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#endif // TORCH_BLADE_USE_ROCM
#endif // TORCH_BLADE_BUILD_WITH_CUDA

#include "tensorflow/compiler/mlir/xla/ral/ral_api.h"

#include "common_utils/utils.h"

#ifdef TORCH_BLADE_USE_ROCM
namespace c10 {
namespace cuda {
const auto& current_device = ::c10::hip::current_device;
const auto& getCurrentCUDAStream = ::c10::hip::getCurrentHIPStream;
namespace CUDACachingAllocator = ::c10::hip::HIPCachingAllocator;
} // namespace cuda
} // namespace c10
#endif // TORCH_BLADE_USE_ROCM

namespace torch {
namespace blade {

class RalAllocator : public tao::ral::Allocator {
 public:
  using buffer_t = tao::ral::buffer_t;
  using alloc_t = tao::ral::alloc_t;
  using dealloc_t = tao::ral::dealloc_t;
  RalAllocator(alloc_t alloc_func, dealloc_t dealloc_func)
      : alloc_func_(alloc_func), dealloc_func_(dealloc_func) {}

  buffer_t alloc(size_t bytes) {
    return alloc_func_(bytes);
  }

  void dealloc(buffer_t buffer) {
    dealloc_func_(buffer);
  }

 private:
  alloc_t alloc_func_;
  dealloc_t dealloc_func_;
};

// Check if every tensor in a list of tensors matches the current
// device.
bool RalContext::CheckCurrentDevice(
    const torch::List<torch::Tensor>& inputs) const {
#ifdef TORCH_BLADE_BUILD_WITH_CUDA
  TORCH_CHECK(gpu_device_ == c10::cuda::current_device());
  // TODO(gty): Refactor this function together with the one defined in TensorRT
  // Engine Context
  if (inputs.empty()) {
    return true;
  }

  // TODO: to support cpu only torch
  torch::Device cur_cuda_device =
      torch::Device(torch::kCUDA, c10::cuda::current_device());
  if (input_dev_.empty()) {
    // NB: to be compatiable with version < 0.0.3
    return std::all_of(inputs.begin(), inputs.end(), [&](torch::Tensor t) {
      return t.device() == cur_cuda_device;
    });
  } else {
    TORCH_CHECK(input_dev_.size() == inputs.size());
    for (size_t k = 0; k < input_dev_.size(); ++k) {
      torch::Tensor inp = inputs[k];
      if (input_dev_[k] == "gpu" && inp.device() != cur_cuda_device) {
        return false;
      }
    }
    return true;
  }
#endif // TORCH_BLADE_BUILD_WITH_CUDA
  return true;
}

std::tuple<void*, void*> RalContext::LoadEngine(
    const std::string& ral_engine_bytes) {
  // Also had tried with shm_fs, however, dlopen tao_lib is not always
  // successful.
  lib_tmpf_.WriteBytesToFile(ral_engine_bytes);
  std::string filename = lib_tmpf_.GetFilename();
  void* tao_lib = dlopen(filename.c_str(), RTLD_NOW | RTLD_LOCAL);
  TORCH_CHECK(tao_lib, "Fail to open ral engine");

  void* func_handle = dlsym(tao_lib, kMlirLoweredEntry);
  TORCH_CHECK(func_handle, "Fail to find kMlirLoweredEntry");
  return std::make_tuple(tao_lib, func_handle);
}

RalContext::~RalContext() {
  if (tao_lib_ != nullptr) {
    dlclose(tao_lib_);
  }
}

RalContext::RalContext(
    const std::string& ral_engine_bytes,
    const std::string& ral_const_bytes,
    const std::string& input_type_spec_str,
    const std::string& output_type_spec_str,
    const std::string& input_dev_str,
    const std::string& output_dev_str)
    : input_type_spec_(
          std::move(ShapeTypeSpec::Deserialize(input_type_spec_str))),
      output_type_spec_(
          std::move(ShapeTypeSpec::Deserialize(output_type_spec_str))) {
  input_dev_ = split(input_dev_str, ",");
  output_dev_ = split(output_dev_str, ",");

  meta_tmpf_.WriteBytesToFile(ral_const_bytes);
  default_opt_.metadata_file_path = meta_tmpf_.GetFilename();
  default_opt_.cache_workspace_mem_across_execution = true;
  cpu_opt_.cpu_allocator.reset(new RalAllocator(c10::alloc_cpu, c10::free_cpu));

#ifdef TORCH_BLADE_BUILD_WITH_CUDA
  at::globalContext().lazyInitCUDA();
  gpu_device_ = c10::cuda::current_device();
#else
  ral_ctx_ = tao::ral::cpu::MakeBaseCpuContext(default_opt_, cpu_opt_);
#endif // TORCH_BLADE_BUILD_WITH_CUDA

  void* func_handle = nullptr;
  std::tie(tao_lib_, func_handle) = LoadEngine(ral_engine_bytes);

  using func_t = void (*)(void**);
  entry_func_ = (func_t)func_handle;

  CHECK_NOTNULL(entry_func_);
}

torch::List<torch::Tensor> RalContext::PreProcessInputs(
    const torch::List<torch::Tensor>& inputs) const {
  // TODO: we currently only support inputs on the same device as tensorrt
  TORCH_CHECK(CheckCurrentDevice(inputs));

  torch::List<torch::Tensor> contiguous_inputs;
  for (torch::Tensor inp_tensor : inputs) {
    // make sure the input is in contiguous layout
    auto contiguous_tensor = inp_tensor.contiguous();
    contiguous_inputs.push_back(contiguous_tensor);
  }
  return contiguous_inputs;
}

void RalContext::BindingInputs(
    const torch::List<torch::Tensor>& inputs,
    tao::ral::ExecutionContext& exec_ctx) const {
  for (size_t idx = 0; idx < inputs.size(); ++idx) {
    torch::Tensor inp = inputs[idx];
    const auto& shape = inp.sizes();
    exec_ctx.bindInput(idx, inp.data_ptr(), shape.vec());
  }
}

bool IsEmptyTensor(const tao::ral::buffer_shape_t& shape) {
  for (int64_t dim : shape) {
    if (dim == 0)
      return true;
  }
  return false;
}

torch::List<torch::Tensor> RalContext::CreateAndBindingOutputs(
    tao::ral::ExecutionContext& exec_ctx) const {
  torch::List<torch::Tensor> outputs;
  const auto& shape_types = output_type_spec_.shape_types();
  auto num_outputs = shape_types.size();
  outputs.reserve(num_outputs);
  std::vector<std::unique_ptr<tao::ral::OutputBufferWrapper>> out_bufs(
      num_outputs);
  for (size_t idx = 0; idx < num_outputs; ++idx) {
    auto& out_buf = out_bufs[idx];
    // Note: Ral has memory allocator that allocate memory each time forward.
    // So it's thread-safe to reuse the underline memory.
    exec_ctx.bindOutput(idx, &out_buf);

    const auto& shape_type = shape_types[idx];
    auto scalar_type = shape_type.type;

#ifdef TORCH_BLADE_BUILD_WITH_CUDA
    torch::DeviceType dev_type =
        torch::kCUDA; // default compatiable with torch_blade version < 0.0.3
    if (output_dev_.size() > 0) {
      dev_type = (output_dev_[idx] == "gpu") ? torch::kCUDA : torch::kCPU;
    }
#else
    torch::DeviceType dev_type = torch::kCPU;
#endif // TORCH_BLADE_BUILD_WITH_CUDA

    auto option = torch::device(dev_type)
                      .dtype(scalar_type)
                      .memory_format(torch::MemoryFormat::Contiguous);
    torch::Tensor out_tensor = IsEmptyTensor(out_buf->shape())
        ? torch::zeros(out_buf->shape(), option)
        : torch::from_blob(
              const_cast<void*>(out_buf->data()), out_buf->shape(), option)
              .clone();
    outputs.push_back(out_tensor);
  }
  return outputs;
}

#ifdef TORCH_BLADE_BUILD_WITH_CUDA
tao::ral::BaseContext* RalContext::LoadCache() {
  TORCH_CHECK(gpu_device_ == c10::cuda::current_device())
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(gpu_device_);

  // TODO: take care of the duplicated const
  // which currently is managed per context
  tao::ral::gpu::BaseCudaContextOption gpu_opt;
  gpu_opt.device_ordinal = gpu_device_;
  gpu_opt.use_stream_executor = true;
  gpu_opt.gpu_allocator.reset(new RalAllocator(
      c10::cuda::CUDACachingAllocator::raw_alloc,
      c10::cuda::CUDACachingAllocator::raw_delete));

  std::lock_guard<std::mutex> guard(mtx_);
  tao::ral::BaseContext* ral_ctx_ptr;
  auto it = ral_ctx_map_.find(stream);
  if (it == ral_ctx_map_.end()) {
    gpu_opt.stream = stream.stream();
    auto ral_ctx =
        tao::ral::gpu::MakeBaseCudaContext(default_opt_, cpu_opt_, gpu_opt);
    ral_ctx_ptr = ral_ctx.get();
    ral_ctx_map_[stream].reset(ral_ctx.release());
  } else {
    ral_ctx_ptr = it->second.get();
  }
  return ral_ctx_ptr;
}
#endif // TORCH_BLADE_BUILD_WITH_CUDA

torch::List<torch::Tensor> RalContext::Forward(
    const torch::List<torch::Tensor>& inputs) {
#ifdef TORCH_BLADE_BUILD_WITH_CUDA
  auto ral_ctx = LoadCache();
  // execution context is per-inference context and thread-safe
  auto exec_ctx =
      tao::ral::MakeExecutionContext<tao::ral::gpu::BaseCudaExecutionContext>(
          ral_ctx);
#else
  auto exec_ctx =
      tao::ral::MakeExecutionContext<tao::ral::cpu::BaseCpuExecutionContext>(
          ral_ctx_.get());
#endif // TORCH_BLADE_BUILD_WITH_CUDA

  auto contiguous_inputs = PreProcessInputs(inputs);
  BindingInputs(contiguous_inputs, *exec_ctx.get());
  auto tao_ral_func_ptr = reinterpret_cast<void*>(&tao_ral_call_impl);

  // execute
  void* ctx_struct[] = {exec_ctx.get(), tao_ral_func_ptr};
  try {
    entry_func_(ctx_struct);
  } catch (std::exception& ex) {
    LOG(ERROR) << ex.what();
    throw ex;
  }

  auto outputs = CreateAndBindingOutputs(*exec_ctx.get());
  return outputs;
}
} // namespace blade
} // namespace torch

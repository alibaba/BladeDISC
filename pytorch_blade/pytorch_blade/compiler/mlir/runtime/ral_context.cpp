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

#include "pytorch_blade/compiler/mlir/runtime/ral_context.h"

#include <dlfcn.h>

#include <c10/core/CPUAllocator.h>
#if PYTORCH_MAJOR_VERSION == 1 && PYTORCH_MINOR_VERSION >= 12
#include <c10/core/impl/alloc_cpu.h>
#endif

#ifdef TORCH_BLADE_BUILD_WITH_CUDA
#ifdef TORCH_BLADE_USE_ROCM
#include <c10/hip/HIPCachingAllocator.h>
#include <c10/hip/HIPFunctions.h>
#else // TORCH_BLADE_USE_ROCM
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#endif // TORCH_BLADE_USE_ROCM
#endif // TORCH_BLADE_BUILD_WITH_CUDA

#include "mlir/ral/ral_api.h"

#include "pytorch_blade/common_utils/utils.h"

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
void RalContext::CheckCurrentDevice(const at::List<at::Tensor>& inputs) {
#ifdef TORCH_BLADE_BUILD_WITH_CUDA
  int64_t gpu_device = LazyInitCurrentDevice();
  // TODO(gty): Refactor this function together with the one defined in TensorRT
  // Engine Context
  if (inputs.empty()) {
    return;
  }

  // TODO: to support cpu only torch
  torch::Device cur_cuda_device = torch::Device(torch::kCUDA, gpu_device);

  auto& inputs_info = engine_state_->inputs;
  TORCH_CHECK(inputs_info.size() == inputs.size());
  for (size_t k = 0; k < inputs.size(); ++k) {
    at::Tensor inp = inputs[k];
    auto device = inputs_info[k].device;
    if (device == "cuda") {
      TORCH_CHECK(
          inp.device() == cur_cuda_device,
          "Input tensor ",
          k,
          " device mismatch. Expect: ",
          cur_cuda_device,
          ", got: ",
          inp.device());
    }
  }
  return;
#endif // TORCH_BLADE_BUILD_WITH_CUDA
  return;
}

std::tuple<void*, void*> RalContext::LoadEngine(
    const std::string& ral_engine_bytes) {
  // Also had tried with shm_fs, however, dlopen tao_lib is not always
  // successful.
  auto is_ok = lib_tmpf_.WriteBytesToFile(ral_engine_bytes);
  TORCH_CHECK(is_ok, "Failed to dump RAL engine to file");
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

RalContext::RalContext(std::shared_ptr<backends::EngineState> state)
    : engine_state_(state) {
  auto is_ok = meta_tmpf_.WriteBytesToFile(state->model_proto);
  TORCH_CHECK(is_ok, "FAiled to dump model proto to file.");
  default_opt_.metadata_file_path = meta_tmpf_.GetFilename();
  default_opt_.cache_workspace_mem_across_execution = true;
  auto torch_allocator = c10::GetAllocator(torch::kCPU);
  TORCH_CHECK(torch_allocator != nullptr);
  auto cpu_alloc = [torch_allocator](size_t n) {
    return torch_allocator->raw_allocate(n);
  };
  auto cpu_delete = [torch_allocator](void* ptr) {
    torch_allocator->raw_deallocate(ptr);
  };
  cpu_opt_.cpu_allocator.reset(new RalAllocator(cpu_alloc, cpu_delete));

#ifdef TORCH_BLADE_BUILD_WITH_CUDA
  at::globalContext().lazyInitCUDA();
#else
  ral_ctx_ = tao::ral::cpu::MakeBaseCpuContext(default_opt_, cpu_opt_);
#endif // TORCH_BLADE_BUILD_WITH_CUDA

  void* func_handle = nullptr;
  std::tie(tao_lib_, func_handle) = LoadEngine(state->engine_bytes);

  using func_t = void (*)(void**);
  entry_func_ = (func_t)func_handle;

  CHECK(entry_func_ != nullptr);
}

at::List<at::Tensor> RalContext::PreProcessInputs(
    const at::List<at::Tensor>& inputs) {
  // TODO: we currently only support inputs on the same device as tensorrt
  CheckCurrentDevice(inputs);

  at::List<at::Tensor> contiguous_inputs;
  for (at::Tensor inp_tensor : inputs) {
    // make sure the input is in contiguous layout
    auto contiguous_tensor = inp_tensor.contiguous();
    contiguous_inputs.push_back(contiguous_tensor);
  }
  return contiguous_inputs;
}

void RalContext::BindingInputs(
    const at::List<at::Tensor>& inputs,
    tao::ral::ExecutionContext& exec_ctx) {
  for (size_t idx = 0; idx < inputs.size(); ++idx) {
    at::Tensor inp = inputs[idx];
    const auto& shape = inp.sizes();
    exec_ctx.bindInput(idx, inp.data_ptr(), shape.vec());
  }
}

inline bool IsEmptyTensor(const tao::ral::buffer_shape_t& shape) {
  return shape.size() > 0 &&
      std::any_of(
             shape.begin(), shape.end(), [](int64_t dim) { return dim == 0; });
}

at::List<at::Tensor> RalContext::CreateAndBindingOutputs(
    tao::ral::ExecutionContext& exec_ctx) {
  at::List<at::Tensor> outputs;

  auto num_outputs = engine_state_->outputs.size();
  outputs.reserve(num_outputs);
  std::vector<std::unique_ptr<tao::ral::OutputBufferWrapper>> out_bufs(
      num_outputs);
  for (size_t idx = 0; idx < num_outputs; ++idx) {
    auto& out_buf = out_bufs[idx];
    // Note: Ral has memory allocator that allocate memory each time forward.
    // So it's thread-safe to reuse the underline memory.
    exec_ctx.bindOutput(idx, &out_buf);

    const auto& output_info = engine_state_->outputs[idx];
    auto scalar_type = output_info.scalar_type;
#ifdef TORCH_BLADE_BUILD_WITH_CUDA
    torch::DeviceType dev_type = torch::kCUDA;
    dev_type = (output_info.device == "cuda") ? torch::kCUDA : torch::kCPU;
#else
    torch::DeviceType dev_type = torch::kCPU;
#endif // TORCH_BLADE_BUILD_WITH_CUDA

    auto option = torch::device(dev_type)
                      .dtype(scalar_type)
                      .memory_format(torch::MemoryFormat::Contiguous);
    at::Tensor out_tensor;
    if (IsEmptyTensor(out_buf->shape())) {
      out_tensor = torch::zeros(out_buf->shape(), option);
    } else if (out_buf->owned()) {
      auto cpu_allocator = c10::GetAllocator(torch::kCPU);
      TORCH_CHECK(cpu_allocator != nullptr);
      std::function<void(void*)> deleter = [cpu_allocator](void* ptr) {
        cpu_allocator->raw_deallocate(ptr);
      };
#ifdef TORCH_BLADE_BUILD_WITH_CUDA
      if (output_info.device == "cuda") {
        deleter = c10::cuda::CUDACachingAllocator::raw_delete;
      }
#endif
      out_tensor = torch::from_blob(
          const_cast<void*>(out_buf->data()),
          out_buf->shape(),
          deleter,
          option);
      out_buf->release();
    } else {
      out_tensor =
          torch::from_blob(
              const_cast<void*>(out_buf->data()), out_buf->shape(), option)
              .clone();
    }
    outputs.push_back(out_tensor);
  }
  return outputs;
}

#ifdef TORCH_BLADE_BUILD_WITH_CUDA
tao::ral::BaseContext* RalContext::LoadCache() {
  int64_t gpu_device = LazyInitCurrentDevice();
  TORCH_CHECK(
      gpu_device >= 0, "expect gpu device id >= 0, but got ", gpu_device);
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(gpu_device);

  // TODO: take care of the duplicated const
  // which currently is managed per context
  tao::ral::gpu::BaseCudaContextOption gpu_opt;
  gpu_opt.device_ordinal = gpu_device;
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

// Because weight is loaded by RAL lazily on the first inference. And there's no
// way to to move loaded weight to another devices in RAL currently. So we need
// to make sure no change on current device during inference. So the reasonable
// restriction is to deny device change during inferences but allow it before
// the first inference.
int64_t RalContext::LazyInitCurrentDevice() {
  int64_t cur_device = c10::cuda::current_device();
  int64_t prev_device = NULL_GPU_DEVICE;
  bool success = gpu_device_.compare_exchange_strong(prev_device, cur_device);
  if (!success) {
    TORCH_CHECK(
        prev_device == cur_device,
        "Device changed during inference. Please do NOT change CUDA "
        "current device during inference.");
  }
  TORCH_CHECK(gpu_device_ != NULL_GPU_DEVICE);
  return cur_device;
}
#endif // TORCH_BLADE_BUILD_WITH_CUDA

at::List<at::Tensor> RalContext::Execute(const at::List<at::Tensor>& inputs) {
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

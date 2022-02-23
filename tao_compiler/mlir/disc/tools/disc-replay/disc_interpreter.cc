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

#include "tensorflow/compiler/mlir/disc/tools/disc-replay/disc_interpreter.h"

#include <dlfcn.h>

namespace replay {

#if GOOGLE_CUDA
using ::stream_executor::gpu::GpuDevicePtr;
#endif

DiscInterpreter::DiscInterpreter() {
  ral_func_ptr_ = reinterpret_cast<void*>(&tao_ral_call_impl);
}

tensorflow::Status DiscInterpreter::Compile(
    tensorflow::tao::TaoCompilerInput& input, CompiledResult& result) {
  auto env = tensorflow::Env::Default();
  std::string tmp_file;
  env->LocalTempFilename(&tmp_file);

  // compile input proto to executable file
  tensorflow::DeviceType device_type(input.options().device_type());
  auto* compiler_wrapper =
      tensorflow::tao::CompilerBase::GetCompilerForDevice(device_type)
          .ConsumeValueOrDie();
  TF_RETURN_IF_ERROR(compiler_wrapper->Compile(input, tmp_file));
  result.output_fname = tmp_file + ".so";
  result.meta_fname = tmp_file + ".so.pbtxt";
  result.input = input;
  TF_RETURN_IF_ERROR(GetEntryFunc(result.output_fname, result.entry_func));
  InitExecCUDAContext(result.meta_fname);

  return tensorflow::Status::OK();
}

tensorflow::Status DiscInterpreter::BindInputs(
    const std::vector<tensorflow::Tensor>& tensors,
    const std::vector<std::string> placements,
    tao::ral::ExecutionContext& exec_ctx) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto t = tensors[i];
    std::vector<int64_t> shape;
    for (size_t dim_i = 0; dim_i < t.dims(); ++dim_i) {
      shape.push_back(t.dim_size(dim_i));
    }
#if GOOGLE_CUDA
    if (placements[i] == "cpu") {
      exec_ctx.bindInput(i, t.data(), shape);
    } else {
      void* d_addr = nullptr;
      auto result = cuMemAlloc((GpuDevicePtr*)&d_addr, t.TotalBytes());
      if (result != CUDA_SUCCESS) {
        return tensorflow::errors::Internal("cuda memory alloc failed");
      }
      result = cuMemcpyHtoD((GpuDevicePtr)d_addr, t.data(), t.TotalBytes());
      if (result != CUDA_SUCCESS) {
        return tensorflow::errors::Internal("cuda memcpy H2D failed");
      }
      cuda_input_ptrs_.push_back(d_addr);
      exec_ctx.bindInput(i, d_addr, shape);
    }
#else
    if (placements[i] != "cpu") {
      return tensorflow::errors::Internal(
          "unexpected input placement, only host tag for CPU only build");
    } else {
      exec_ctx.bindInput(i, t.data(), shape);
    }
#endif
  }
  return tensorflow::Status::OK();
}

tensorflow::Status DiscInterpreter::Run(
    const CompiledResult& compiled_result,
    const std::vector<tensorflow::Tensor>& tensors,
    const std::vector<std::string>& placements) {
#if GOOGLE_CUDA
  auto exec_ctx =
      tao::ral::MakeExecutionContext<tao::ral::gpu::BaseCudaExecutionContext>(
          context_.get());
#else
  auto exec_ctx =
      tao::ral::MakeExecutionContext<tao::ral::cpu::BaseCpuExecutionContext>(
          context_.get());
#endif
  TF_RETURN_IF_ERROR(BindInputs(tensors, placements, *exec_ctx.get()));
  void* ctx_struct[] = {exec_ctx.get(), ral_func_ptr_};
  compiled_result.entry_func(ctx_struct);
  return GCCudaMemory(compiled_result, *exec_ctx.get());
}

tensorflow::Status DiscInterpreter::GCCudaMemory(
    const CompiledResult& compiled_result,
    tao::ral::ExecutionContext& exec_ctx) {
  // GC input tensors
  for (auto ptr : cuda_input_ptrs_) {
    CUdeviceptr device_ptr = absl::bit_cast<CUdeviceptr>(ptr);
    auto res = cuMemFree(device_ptr);
    if (res != CUDA_SUCCESS) {
      LOG(ERROR) << "failed to free device memory at " << ptr << res;
      return tensorflow::errors::Internal(
          "deallocate input device memory failed");
    }
  }
  cuda_input_ptrs_.clear();
  // GC output tensors
  auto num_outputs = compiled_result.input.options().output_placements().size();
  for (size_t i = 0; i < num_outputs; ++i) {
    std::unique_ptr<tao::ral::OutputBufferWrapper> ptr;
    exec_ctx.bindOutput(i, &ptr);
    CUdeviceptr device_ptr = absl::bit_cast<CUdeviceptr>(ptr->data());
    auto res = cuMemFree(device_ptr);
    if (res != CUDA_SUCCESS) {
      LOG(ERROR) << "failed to free device memory at " << ptr->data() << res;
      return tensorflow::errors::Internal(
          "deallocate output device memory failed");
    }
  }
  return tensorflow::Status::OK();
}

tensorflow::SubAllocator* CreateGPUMemAllocator() {
  tensorflow::PlatformDeviceId gpu_id(0);
  return new tensorflow::DeviceMemAllocator(
      tensorflow::DeviceIdUtil::ExecutorForPlatformDeviceId(
          tensorflow::GPUMachineManager(), gpu_id)
          .ValueOrDie(),
      gpu_id,
      /*use_unified_memory=*/false, {}, {});
}

void DiscInterpreter::InitExecCUDAContext(const std::string& meta_fname) {
  tao::ral::BaseContextOption opt;
  opt.metadata_file_path = meta_fname;
  opt.cache_workspace_mem_across_execution = true;
  tao::ral::cpu::BaseCpuContextOption cpu_opt;
#if GOOGLE_CUDA
  tao::ral::gpu::BaseCudaContextOption gpu_opt;
  gpu_opt.use_stream_executor = true;
  // using BFCAllocator to achieve better allocation performance
  tensorflow::GPUOptions options;
  options.set_allow_growth(true);
  options.set_force_gpu_compatible(true);
  gpu_opt.gpu_allocator.reset(
      new tao::ral::TFAllocatorWrapper(new tensorflow::GPUBFCAllocator(
          std::move(CreateGPUMemAllocator()), 16106127360ull /*15GB*/, options,
          "GPU_0_bfc")));
  context_ = tao::ral::gpu::MakeBaseCudaContext(opt, cpu_opt, gpu_opt);
#else
  context_ = tao::ral::cpu::MakeBaseCpuContext(opt, cpu_opt);
#endif
}

tensorflow::Status DiscInterpreter::GetEntryFunc(
    const std::string& exectuable_fname, func_t& entry_func) {
  void* func_handle = dlopen(exectuable_fname.c_str(), RTLD_NOW);
  if (!func_handle) {
    std::string msg = "fail to open compiled .so file with error: ";
    absl::StrAppend(&msg, dlerror());
    return tensorflow::errors::Internal(msg);
  }

  void* entry_func_ptr = dlsym(func_handle, "main");
  if (!entry_func_ptr) {
    return tensorflow::errors::Internal("fail to find the main");
  }
  entry_func = (func_t)entry_func_ptr;
  CHECK_NOTNULL(entry_func);
  return tensorflow::Status::OK();
}

}  //  namespace replay

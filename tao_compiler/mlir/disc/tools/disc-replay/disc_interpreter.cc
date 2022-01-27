#include <dlfcn.h>

#include "tensorflow/compiler/mlir/disc/tools/disc-replay/disc_interpreter.h"

namespace replay {

DiscInterpreter::DiscInterpreter() {
  ral_func_ptr_ = reinterpret_cast<void*>(&tao_ral_call_impl);
}


tensorflow::Status DiscInterpreter::Run(const std::string& input_fname, const std::string& record_fname) {
  // 1. load tao compilation input pb and compile into executable
  // 2. load ReplayRecord file with protobuf format
  // 3. Run the executable with binding the ReplayRecord tensors.
  std::string output_fname = input_fname + ".output";
  auto s = LoadAndCompileToExecutable(input_fname, output_fname);
  if (!s.ok()) {
    return tensorflow::errors::Internal("Compile to executable failed");
  }
}

tensorflow::Status DiscInterpreter::LoadAndCompileToExecutable(const std::string& input_fname, const std::string& output_fname) {
  // load proto into tao input
  tensorflow::tao::TaoCompilerInput input;
  TF_RETURN_IF_ERROR(ReadBinaryProto(tensorflow::Env::Default(), input_fname, &input));

  // compile input proto to executable file
  tensorflow::DeviceType device_type(input.options().device_type());
  auto* compiler_wrapper =
      tensorflow::tao::CompilerBase::GetCompilerForDevice(device_type).ConsumeValueOrDie();
  TF_RETURN_IF_ERROR(compiler_wrapper->Compile(input, output_fname));
  return tensorflow::Status::OK();
}

std::unique_ptr<tao::ral::gpu::BaseCudaExecutionContext> DiscInterpreter::GetExecCUDAContext(const std::string& executable_fname) {
  tao::ral::BaseContextOption opt;
  opt.metadata_file_path = executable_fname + ".pbtxt";
  opt.cache_workspace_mem_across_execution = true;
  tao::ral::cpu::BaseCpuContextOption cpu_opt;
  tao::ral::gpu::BaseCudaContextOption gpu_opt;
  gpu_opt.use_stream_executor = true;
  auto context = tao::ral::gpu::MakeBaseCudaContext(opt, cpu_opt, gpu_opt);
  return tao::ral::MakeExecutionContext<
            tao::ral::gpu::BaseCudaExecutionContext>(context.get());
}

tensorflow::Status DiscInterpreter::GetEntryFunc(const std::string& exectuable_fname, func_t* entry_func) {
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
  *entry_func = (func_t)entry_func_ptr;
  return tensorflow::Status::OK();
}

tensorflow::Status DiscInterpreter::RunExecutable(std::vector<tensorflow::Tensor> tensors, const std::string& executable_fname) {
  func_t entry_func;
  TF_RETURN_IF_ERROR(GetEntryFunc(executable_fname, &entry_func));
  auto exec_ctx = GetExecCUDAContext(executable_fname);

  void* ctx_struct[] = {exec_ctx.get(), ral_func_ptr_};
  void** args = (void**)(&ctx_struct);

  entry_func(args);
}

} //  namespace replay

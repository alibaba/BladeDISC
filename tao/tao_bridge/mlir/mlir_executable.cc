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

#include "tao_bridge/mlir/mlir_executable.h"

#include <dlfcn.h>

#ifdef PLATFORM_ALIBABA
#include "tao_bridge/tao_launch_op/tao_launch.h"
#else
#include "tao_bridge/kernels/disc_launch.h"
#endif  // PLATFORM_ALIBABA

#include "mlir/ral/ral_api.h"

namespace tensorflow {
namespace tao {

MlirExecutable::MlirExecutable(const string& compiled_result_file,
                               const string& target_device)
    : Executable(compiled_result_file),
      target_device_(target_device),
      dso_handle_(nullptr) {}

MlirExecutable::~MlirExecutable() {
  if (dso_handle_ != nullptr) {
    int ret = dlclose(dso_handle_);
    char* err_msg = dlerror();
    if (ret != 0) {
      LOG(ERROR) << "error when close dso handle to " << dso_file_
                 << ", error code: " << ret << ", error message: " << err_msg;
    }
  }
}

void MlirExecutable::DumpToFile(const std::string& filename) const {
  TaoCompilerResult result(tao_compiler_result());
  *result.mutable_target_device() = target_device();
  *result.mutable_mlir()->mutable_so_lib_filename() = filename + ".so";
  *result.mutable_mlir()->mutable_const_proto_filename() = filename + ".so.pb";
  if (tao_compiler_result().mlir().so_lib_filename() !=
      result.mlir().so_lib_filename()) {
    TF_CHECK_OK(tensorflow::Env::Default()->CopyFile(
        tao_compiler_result().mlir().so_lib_filename(),
        result.mlir().so_lib_filename()));
  }
  if (tao_compiler_result().mlir().const_proto_filename() !=
      result.mlir().const_proto_filename()) {
    TF_CHECK_OK(tensorflow::Env::Default()->CopyFile(
        tao_compiler_result().mlir().const_proto_filename(),
        result.mlir().const_proto_filename()));
  }
  CHECK(WriteTextProto(tensorflow::Env::Default(), filename, result).ok());
}

Status MlirExecutable::PreRunProcess(const ExecutableRunOptions& options,
                                     BufferAllocations& allocations,
                                     std::vector<Tensor>& output_tensors) {
  return Status::OK();
}

Status MlirExecutable::PostRunProcess(const ExecutableRunOptions& options,
                                      BufferAllocations& allocations,
                                      std::vector<Tensor>& output_tensors) {
  return Status::OK();
}

std::string MlirExecutable::target_device() const { return target_device_; }

Status MlirExecutable::InitImpl(const TaoCompilerResult* result) {
  if (!result->has_mlir()) {
    return errors::Internal("Compilation result has no mlir field: ",
                            compiled_result_file());
  }
  dso_file_ = result->mlir().so_lib_filename();
  if (dso_file_.empty()) {
    return errors::Internal("Empty so_lib_filename in: ",
                            compiled_result_file());
  }

  VLOG(1) << "load compiled DSO from: " << dso_file_;
  dso_handle_ = dlopen(dso_file_.c_str(), RTLD_NOW | RTLD_LOCAL);
  char* err_msg = dlerror();
  if (dso_handle_ == nullptr) {
    return errors::Internal("load DSO failed, file: ", dso_file_,
                            ", error: ", err_msg);
  }

  void* func_handle = dlsym(dso_handle_, kMlirLoweredEntry);
  err_msg = dlerror();
  if (func_handle == nullptr) {
    return errors::Internal("look up symble failed, file: ", dso_file_,
                            ", symble: ", kMlirLoweredEntry,
                            ", error: ", err_msg);
  }

  entry_func_ = (MLIR_FUNC_T)func_handle;

  RalTfContextOptions opts;
  opts.metadata_file_path = result->mlir().const_proto_filename();
  mutable_ral_context().reset(new RalTfContext(opts));
  return Status::OK();
}

Status MlirExecutable::RunImpl(const ExecutableRunOptions& options,
                               BufferAllocations& allocations) {
  auto* ral_ctx = ral_context();
  auto exec_ctx =
      ::tao::ral::MakeExecutionContext<RalTfExecutionContext>(ral_ctx);
  exec_ctx->setOpContext(options.ctx());

  void* ctx_struct[] = {exec_ctx.get(), (void*)tao_ral_call_impl};
  entry_func_(ctx_struct);
  return Status::OK();
}

#ifndef TAO_CPU_ONLY
// TODO(kevin.zwy): change to use TAO_HAS_GPU instead of using TAO_CPU_ONLY
static std::unique_ptr<Executable> NewMlirGpuExecutable(
    const string& compiled_result_file) {
  return std::unique_ptr<Executable>(
      new MlirExecutable(compiled_result_file, "MLIR_GPU"));
}
TAO_REGISTER_EXECUTABLE("MLIR_GPU", NewMlirGpuExecutable);
#ifdef PLATFORM_ALIBABA
REGISTER_TAO_MLIR_LAUNCH_KERNEL(DEVICE_GPU);
#else
REGISTER_DISC_LAUNCH_KERNEL(DEVICE_GPU);
#endif  // PLATFORM_ALIBABA
#endif  // TAO_CPU_ONLY

static std::unique_ptr<Executable> NewMlirCpuExecutable(
    const string& compiled_result_file) {
  return std::unique_ptr<Executable>(
      new MlirExecutable(compiled_result_file, "MLIR_CPU"));
}
TAO_REGISTER_EXECUTABLE("MLIR_CPU", NewMlirCpuExecutable);
#ifdef PLATFORM_ALIBABA
REGISTER_TAO_MLIR_LAUNCH_KERNEL(DEVICE_CPU);
#else
REGISTER_DISC_LAUNCH_KERNEL(DEVICE_CPU);
#endif  // PLATFORM_ALIBABA

}  // namespace tao
}  // namespace tensorflow

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

#include "tao_bridge/kernels/launch_base.h"

namespace tensorflow {
namespace tao {
LaunchBase::LaunchBase(OpKernelConstruction* ctx)
    : AsyncOpKernel(ctx),
      ctx_(ctx),
      constants_(ConstantsVector()),
      fixed_shapes_(FixedShapesVector()),
      host_args_(HostArgsVector()),
      resources_(ResourcesVector()) {
  OP_REQUIRES_OK(ctx, PlatformInfoFromContext(ctx_, &platform_info_));
  OP_REQUIRES_OK(ctx, DeviceUUIDFromContext(&device_uuid_));
}

Status LaunchBase::DeviceUUIDFromContext(std::string* result_uuid) {
  if (ctx_->device_type() != DeviceType(DEVICE_GPU)) {
    return Status::OK();
  }

  std::string bus_id = ctx_->device()
                           ->tensorflow_gpu_device_info()
                           ->stream->parent()
                           ->GetDeviceDescription()
                           .pci_bus_id();

  *result_uuid = cuda_utils::GetGpuDeviceUUID(bus_id);
  return Status::OK();
}

Status LaunchBase::RunExecutable(Executable* executable,
                                 ExecutableRunOptions& options,
                                 TaoCompileFuncCallInfo* call_info) {
  TaoCompInfoCollector::Get().SetCallTimestamp(call_info,
                                               TIME_EXEC_BIN_RUN_BEGIN);
  auto status = executable->Run(options);
  TaoCompInfoCollector::Get().SetCallTimestamp(call_info,
                                               TIME_EXEC_BIN_RUN_END);
  return status;
}

// OP_REQUIRES_OK_RETURN is the same as OP_REQUIRES_OK except that
// in error case, it returns RET instead of void.
#define OP_REQUIRES_OK_RETURN(CTX, RET, ...)                \
  do {                                                      \
    ::tensorflow::Status _s(__VA_ARGS__);                   \
    if (!TF_PREDICT_TRUE(_s.ok())) {                        \
      (CTX)->CtxFailureWithWarning(__FILE__, __LINE__, _s); \
      return RET;                                           \
    }                                                       \
  } while (0)

// Helper static functions to construct parameters for
// XlaLocalLaunchBase constructor from OpKernelConstruction.
std::vector<int> LaunchBase::ConstantsVector() {
  DataTypeVector constant_types;
  OP_REQUIRES_OK_RETURN(ctx_, std::vector<int>(),
                        ctx_->GetAttr("Tconstants", &constant_types));
  std::vector<int> constants(constant_types.size());
  std::iota(constants.begin(), constants.end(), 0);
  return constants;
}

std::vector<int> LaunchBase::FixedShapesVector() {
  std::vector<int> fixed_shaped;
  if (ctx_->HasAttr("Tfixedshapes")) {
    DataTypeVector constant_types;
    OP_REQUIRES_OK_RETURN(ctx_, std::vector<int>(),
                          ctx_->GetAttr("Tconstants", &constant_types));

    DataTypeVector fixed_shaped_types;
    OP_REQUIRES_OK_RETURN(ctx_, std::vector<int>(),
                          ctx_->GetAttr("Tfixedshapes", &fixed_shaped_types));

    fixed_shaped.resize(fixed_shaped_types.size());
    std::iota(fixed_shaped.begin(), fixed_shaped.end(), constant_types.size());
  }
  return fixed_shaped;
}

std::vector<int> LaunchBase::HostArgsVector() {
  std::vector<int> host_args;
  if (ctx_->HasAttr("Thostargs") && ctx_->HasAttr("Tfixedshapes")) {
    DataTypeVector constant_types;
    OP_REQUIRES_OK_RETURN(ctx_, std::vector<int>(),
                          ctx_->GetAttr("Tconstants", &constant_types));

    DataTypeVector fixed_shaped_types;
    OP_REQUIRES_OK_RETURN(ctx_, std::vector<int>(),
                          ctx_->GetAttr("Tfixedshapes", &fixed_shaped_types));

    DataTypeVector host_arg_types;
    OP_REQUIRES_OK_RETURN(ctx_, std::vector<int>(),
                          ctx_->GetAttr("Thostargs", &host_arg_types));

    host_args.resize(host_arg_types.size());
    std::iota(host_args.begin(), host_args.end(),
              constant_types.size() + fixed_shaped_types.size());
  }
  return host_args;
}

std::vector<int> LaunchBase::ResourcesVector() {
  DataTypeVector constant_types;
  OP_REQUIRES_OK_RETURN(ctx_, std::vector<int>(),
                        ctx_->GetAttr("Tconstants", &constant_types));

  int num_fixed_shaped = 0;
  if (ctx_->HasAttr("Tfixedshapes")) {
    // only for MlirLaunchOp
    DataTypeVector fixed_shaped_types;
    OP_REQUIRES_OK_RETURN(ctx_, std::vector<int>(),
                          ctx_->GetAttr("Tfixedshapes", &fixed_shaped_types));
    num_fixed_shaped = fixed_shaped_types.size();
  }

  DataTypeVector arg_types;
  OP_REQUIRES_OK_RETURN(ctx_, std::vector<int>(),
                        ctx_->GetAttr("Targs", &arg_types));

  int num_resources = -1;
  OP_REQUIRES_OK_RETURN(ctx_, std::vector<int>(),
                        ctx_->GetAttr("Nresources", &num_resources));

  std::vector<int> resources(num_resources);
  std::iota(resources.begin(), resources.end(),
            constant_types.size() + num_fixed_shaped + arg_types.size());
  return resources;
}

NameAttrList LaunchBase::FunctionAttr(const char* const attr) {
  const NameAttrList* func = nullptr;
  OP_REQUIRES_OK_RETURN(ctx_, NameAttrList(), ctx_->GetAttr(attr, &func));
  return *func;
}

bool LaunchBase::BoolAttr(const char* const attr) {
  bool ret;
  OP_REQUIRES_OK_RETURN(ctx_, false, ctx_->GetAttr(attr, &ret));
  return ret;
}

Tensor LaunchBase::ToCpu(OpKernelContext* ctx, Tensor t, MemoryType mem_type) {
  if (HOST_MEMORY == mem_type) return t;

  AllocatorAttributes alloc_attr;
  auto to_ptr = [](const Tensor& tensor) {
    return const_cast<void*>(
        static_cast<const void*>(tensor.tensor_data().data()));
  };
  auto stream = ctx->op_device_context()->stream();
  Tensor cpu_tensor;
  alloc_attr.set_on_host(true);
  ctx->allocate_temp(t.dtype(), t.shape(), &cpu_tensor, alloc_attr);
  stream->ThenMemcpy(to_ptr(cpu_tensor), se::DeviceMemoryBase(to_ptr(t)),
                     t.TotalBytes());
  stream->BlockHostUntilDone();
  return cpu_tensor;
}

#undef OP_REQUIRES_OK_RETURN

Status LaunchBase::EnsureFunctionHandle(
    OpKernelContext* ctx, const NameAttrList& func,
    FunctionLibraryRuntime::Handle* handle) {
  if (*handle != kInvalidHandle) {
    return Status::OK();
  }
  VLOG(2) << "Constructing function handle " << func.name();
  FunctionLibraryRuntime* lib = ctx->function_library();
  if (lib == nullptr) {
    return tensorflow::errors::Internal("Context function library is null");
  }
  tensorflow::FunctionLibraryRuntime::InstantiateOptions inst_ops;
  inst_ops.target = ctx->device()->name();
  auto status =
      lib->Instantiate(func.name(), AttrSlice(&func.attr()), inst_ops, handle);
  if (!status.ok()) {
    LOG(ERROR) << " Instantiating native function " << func.name()
               << " failed!";
  }
  return status;
}

Status LaunchBase::PrepareInputsCpu(OpKernelContext* ctx,
                                    std::vector<Tensor>* inputs_ptr) {
  CHECK(inputs_ptr != nullptr);
  auto& inputs = *inputs_ptr;
  CHECK_EQ(ctx->op_device_context(), nullptr);
  for (int i = 0; i < ctx->num_inputs(); ++i) {
    auto& input_tensor = ctx->input(i);
    inputs.push_back(input_tensor);
  }
  return Status::OK();
}

Status LaunchBase::SyncInputsMem(OpKernelContext* ctx,
                                 std::vector<Tensor>* inputs_ptr,
                                 bool check_block) {
  CHECK(inputs_ptr != nullptr);
  bool need_block = false;
  auto& inputs = *inputs_ptr;
  for (int i = 0; i < ctx->num_inputs(); ++i) {
    auto& input_tensor = ctx->input(i);
    auto input_memory_type = ctx->input_memory_type(i);
    auto func_memory_type = MTypeFromDType(ctx->input_dtype(i));
    if (input_memory_type != func_memory_type) {
      need_block = true;
      VLOG(2) << "In MemType mismatch #" << i;
      CHECK_NE(ctx->op_device_context(), nullptr);
      auto stream = ctx->op_device_context()->stream();
      Tensor func_tensor;
      AllocatorAttributes alloc_attr;
      auto to_ptr = [](const Tensor& tensor) {
        return const_cast<void*>(
            static_cast<const void*>(tensor.tensor_data().data()));
      };
      if (input_memory_type == DEVICE_MEMORY) {
        // copy from DEVICE to HOST
        VLOG(2) << "\tDEVICE to Host";
        alloc_attr.set_on_host(true);
        ctx->allocate_temp(input_tensor.dtype(), input_tensor.shape(),
                           &func_tensor, alloc_attr);
        stream->ThenMemcpy(to_ptr(func_tensor),
                           se::DeviceMemoryBase(to_ptr(input_tensor)),
                           input_tensor.TotalBytes());
      } else {
        CHECK(HOST_MEMORY == input_memory_type);
        // copy from HOST to DEVICE
        VLOG(2) << "\tHOST to DEVICE";
        ctx->allocate_temp(input_tensor.dtype(), input_tensor.shape(),
                           &func_tensor);
        se::DeviceMemoryBase func_mem(to_ptr(func_tensor));
        stream->ThenMemcpy(&func_mem, to_ptr(input_tensor),
                           input_tensor.TotalBytes());
      }
      inputs.push_back(func_tensor);
    } else {
      inputs.push_back(input_tensor);
    }
  }
  if (need_block && check_block) {
    auto stream = ctx->op_device_context()->stream();
    stream->BlockHostUntilDone();
  }
  return Status::OK();
}

Status LaunchBase::PrepareOutputsCpu(OpKernelContext* ctx,
                                     std::vector<Tensor>* outputs) {
  CHECK_EQ(ctx->op_device_context(), nullptr);
  for (int i = 0; i < ctx->num_outputs(); ++i) {
    auto& output_tensor = outputs->at(i);
    ctx->set_output(i, outputs->at(i));
  }
  return Status::OK();
}

Status LaunchBase::SyncOutputsMem(OpKernelContext* ctx,
                                  std::vector<Tensor>* outputs,
                                  bool check_block) {
  bool need_block = false;
  for (int i = 0; i < ctx->num_outputs(); ++i) {
    auto& output_tensor = outputs->at(i);
    auto output_memory_type = ctx->output_memory_type(i);
    auto func_memory_type = MTypeFromDType(ctx->expected_output_dtype(i));
    if (output_memory_type != func_memory_type) {
      VLOG(2) << "Out MemType mismatch #" << i;
      CHECK_NE(ctx->op_device_context(), nullptr);
      need_block = true;
      auto stream = ctx->op_device_context()->stream();
      Tensor* func_tensor = nullptr;
      auto to_ptr = [](const Tensor& tensor) {
        return const_cast<void*>(
            static_cast<const void*>(tensor.tensor_data().data()));
      };
      if (func_memory_type == DEVICE_MEMORY) {
        // copy from DEVICE to HOST
        VLOG(2) << "\tDEVICE to Host";
        TF_RETURN_IF_ERROR(
            ctx->allocate_output(i, output_tensor.shape(), &func_tensor));
        stream->ThenMemcpy(to_ptr(*func_tensor),
                           se::DeviceMemoryBase(to_ptr(output_tensor)),
                           output_tensor.TotalBytes());
      } else {
        CHECK(HOST_MEMORY == func_memory_type);
        // copy from HOST to DEVICE
        VLOG(2) << "\tHOST to DEVICE";
        TF_RETURN_IF_ERROR(
            ctx->allocate_output(i, output_tensor.shape(), &func_tensor));
        se::DeviceMemoryBase func_mem(to_ptr(*func_tensor));
        stream->ThenMemcpy(&func_mem, to_ptr(output_tensor),
                           output_tensor.TotalBytes());
      }
    } else {
      ctx->set_output(i, outputs->at(i));
    }
  }
  if (need_block && check_block) {
    auto stream = ctx->op_device_context()->stream();
    stream->BlockHostUntilDone();
  }
  return Status::OK();
}

std::map<int, OptionalTensor> LaunchBase::SnapshotResourceVariables(
    OpKernelContext* ctx, absl::Span<const int> variables) {
  std::map<int, OptionalTensor> snapshot;
  for (int i : variables) {
    Var* variable = nullptr;
    ResourceHandle handle = HandleFromInput(ctx, i);
    OptionalTensor& tensor = snapshot[i];
    if (LookupResource(ctx, handle, &variable).ok()) {
      core::ScopedUnref scoped_unref(variable);
      tf_shared_lock lock(*variable->mu());
      tensor.name = handle.name();
      tensor.present = true;
      tensor.value = *variable->tensor();
    }
  }
  return snapshot;
}

void LaunchBase::printInOuts(OpKernelContext* ctx) {
  VLOG(0) << "num inputs #" << ctx->num_inputs() << ":";
  for (int i = 0; i < ctx->num_inputs(); ++i) {
    VLOG(0) << ">> input tensor #" << i << "@"
            << ctx->input(i).shape().DebugString() << " [";
    Tensor cpu_tensor = ToCpu(ctx, ctx->input(i), ctx->input_memory_type(i));
#if TF_MAJOR_VERSION > 1 || TF_MINOR_VERSION > 12
    VLOG(0) << "\t" << cpu_tensor.DebugString(32) << "\n]";
#else
    VLOG(0) << "\t" << cpu_tensor.DebugString() << "\n]";
#endif
  }

  VLOG(0) << "num outputs #" << ctx->num_outputs() << ":";
  for (int i = 0; i < ctx->num_outputs(); ++i) {
    Tensor* out_tensor = ctx->mutable_output(i);
    if (!out_tensor) {
      VLOG(0) << "missing output tensor\n]";
      continue;
    }
    VLOG(0) << ">> output tensor #" << i << "@"
            << out_tensor->shape().DebugString() << " [";
    Tensor cpu_tensor = ToCpu(ctx, *out_tensor, ctx->output_memory_type(i));
#if TF_MAJOR_VERSION > 1 || TF_MINOR_VERSION > 12
    VLOG(0) << "\t" << cpu_tensor.DebugString(32) << "\n]";
#else
    VLOG(0) << "\t" << cpu_tensor.DebugString() << "\n]";
#endif
  }
}

}  //  namespace tao
}  //  namespace tensorflow
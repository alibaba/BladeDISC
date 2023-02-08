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

#include "tao_bridge/kernels/disc_launch.h"

#include "tensorflow/stream_executor/stream_executor.h"

namespace tensorflow {
namespace tao {

DiscLaunchOp::DiscLaunchOp(OpKernelConstruction* ctx)
    : LaunchBase(ctx),
      mlir_function_(FunctionAttr("mlir_function")),
      mlir_func_handle_(kInvalidHandle) {
  auto* bridge_opts = GetTaoBridgeOptions();
  enable_fallback_ = bridge_opts->tao_launch_enable_fallback;
  verbose_compilation_log_ = bridge_opts->verbose_compilation_log;
  VLOG(1) << "Create DiscLaunchOp, device uuid: [" << DeviceUUIDCtx() << "]";
  mode_ = bridge_opts->tao_launch_async_compilation ? kAsync : kDefault;
  mlir_compile_status_ = Status::OK();
  tick_.reset(new TaoLaunchTick(name()));
}

DiscLaunchOp::~DiscLaunchOp() {
  VLOG(2) << "DiscLaunchOp destroyed";
  VLOG(2) << tick_->Report();
}

void DiscLaunchOp::ComputeAsync(OpKernelContext* ctx,
                                AsyncOpKernel::DoneCallback done) {
  auto helper = new DoneHelper(done);
  tensorflow::core::ScopedUnref sc(helper);
  bool executed = false;
  auto status = CompileAndRunMlir(ctx, helper, &executed);
  if (!status.ok() || !executed) {
    if (!status.ok() && err_msg_print_counter_ < 5) {
      ++err_msg_print_counter_;
      LOG(WARNING) << "cluster: " << name() << " fallbacks to TF as "
                   << status.error_message();
    }
    if (enable_fallback_) {
      VLOG(2) << "cluster: " << name() << " fallback to TF.";
      OP_REQUIRES_OK_ASYNC(ctx,
                           ExecuteFunction<GpuTFProfiler>(
                               mlir_function_, &mlir_func_handle_, ctx, helper),
                           done);
    } else if (!status.ok()) {
      OP_REQUIRES_OK_ASYNC(ctx, status, done);
    } else {
      OP_REQUIRES_OK_ASYNC(
          ctx,
          errors::Internal("Async compilation is disabled when "
                           "TAO_ENABLE_FALLBACK is false."),
          done);
    }
  } else if (GetTaoBridgeOptions()->disc_debug_mode) {
    VLOG(0) << "print ins/outs";
    printInOuts(ctx);
  }
}

Status DiscLaunchOp::CompileAndRunMlir(OpKernelContext* ctx, DoneHelper* helper,
                                       bool* executed) {
  *executed = false;
  std::map<int, OptionalTensor> variables;
  TaoProfileStat* stat;
  Executable* mlir_executable = nullptr;

  VLOG(1) << "Run MlirExecutable for node: " << name();
  if (VLOG_IS_ON(1)) {
    VLOG(0) << "Constant (" << ConstantsAttr().size() << "):";
    for (auto&& v : ConstantsAttr()) VLOG(1) << "\tconst_idx: " << v;
    VLOG(0) << "Fix shape (" << FixedShapesAttr().size() << "):";
    for (auto&& v : FixedShapesAttr()) VLOG(1) << "\tfix_shape_idx: " << v;
    VLOG(0) << "Resource (" << ResourcesAttr().size() << "):";
    for (auto&& v : ResourcesAttr()) VLOG(1) << "\tresource_idx: " << v;
  }

  if (GetTaoBridgeOptions()->disc_force_fallback) {
    VLOG(0) << "go fallback path duce `disc_force_fallback` is true";
    return Status::OK();
  }

  static const char* clusters_to_skip = getenv("DISC_SKIP_CLUSTERS");
  if (clusters_to_skip) {
    string refined_name = ";" + name() + ";";
    string clusters_to_skip_str(clusters_to_skip);
    if (clusters_to_skip_str.find(refined_name) != std::string::npos) {
      VLOG(0) << "go fallback path due to cluster " << name()
              << " in DISC_SKIP_CLUSTERS(" << clusters_to_skip << ")";
      return Status::OK();
    }
  }

  static const char* clusters_to_enable = getenv("DISC_ENABLE_CLUSTERS");
  if (clusters_to_enable) {
    string refined_name = ";" + name() + ";";
    string clusters_to_enable_str(clusters_to_enable);
    if (clusters_to_enable_str.find(refined_name) == std::string::npos) {
      VLOG(0) << "go fallback path due to cluster not in" << name()
              << " in DISC_ENABLE_CLUSTERS(" << clusters_to_enable << ")";
      return Status::OK();
    }
  }

  auto call_info = &(helper->call_info);
  TF_RETURN_IF_ERROR(
      CompileToLocalExecutable(ctx, mlir_function_, /* is_mlir */ true,
                               call_info, &variables, &mlir_executable, &stat));

  if (mlir_executable == nullptr) {
    return Status::OK();
  }

  *executed = true;

  ExecutableRunOptions options;
  TF_RETURN_IF_ERROR(PrepareExecutableRunOptions(ctx, ConstantsAttr().size(),
                                                 variables, &options));

  return RunExecutable(mlir_executable, options, call_info);
}

Status DiscLaunchOp::CompileToLocalExecutable(
    OpKernelContext* ctx, const NameAttrList& function, bool is_mlir,
    TaoCompileFuncCallInfo* call_info, std::map<int, OptionalTensor>* variables,
    tao::Executable** executable, TaoProfileStat** stat) {
  TaoCompInfoCollector::Get().SetCallTimestamp(call_info,
                                               TIME_COMPILE_CALL_BEGIN);
  // We store information about the JIT-compiled Result
  // in the ResourceMgr.
  ResourceMgr* rm = ctx->resource_manager();
  if (!rm) {
    return errors::Internal("No resource manager.");
  }

  TaoCompilationCache* cache = nullptr;
  TF_RETURN_IF_ERROR(rm->LookupOrCreate<TaoCompilationCache>(
      rm->default_container(), "tao_cache", &cache,
      [&](TaoCompilationCache** cache) {
        *cache = new TaoCompilationCache(/* async_compilation */ mode_ ==
                                         CompilationMode::kAsync);
        return Status::OK();
      }));
  // Hold the reference to the JIT during evaluation. (We could probably
  // free it sooner because the ResourceMgr will retain a reference, but
  // this is more obviously correct.)
  core::ScopedUnref cache_ref(cache);

  *variables = SnapshotResourceVariables(ctx, ResourcesAttr());
  std::unique_ptr<TaoCompilerInput> input_ptr(new TaoCompilerInput);
  auto& options = *(input_ptr->mutable_options());
  auto flib_def = ctx->function_library();
  auto& device_type_str = PlatformInfoCtx().device_type().type_string();
  if (device_type_str == DEVICE_GPU) {
    *options.mutable_device_type() = "MLIR_GPU";
  } else if (device_type_str == DEVICE_CPU) {
    *options.mutable_device_type() = "MLIR_CPU";
  } else {
    return errors::Internal(
        "Mlir compiler other than CPU/GPU is not implemented yet");
  }
  if (DeviceUUIDCtx().empty()) {
    // if no devicd UUID info, fall back to use device ordinal.
    if (ctx->op_device_context() != nullptr) {
      auto device_ordinal =
          ctx->op_device_context()->stream()->parent()->device_ordinal();
      options.set_device_ordinal(device_ordinal);
    }
  } else if (device_type_str == DEVICE_GPU) {
    // Otherwise we let tao_compiler_main only see the device on which the
    // kernel is placed.
    options.set_device_ordinal(0);
    input_ptr->mutable_env()->insert({"CUDA_VISIBLE_DEVICES", DeviceUUIDCtx()});
  }
  {
    // save this flag into input proto binary. It's useful for us to reproduce
    // compile failures offline.
    const char* envvar_ = getenv("TF_XLA_FLAGS");
    if (envvar_) {
      input_ptr->mutable_env()->insert({"TF_XLA_FLAGS", envvar_});
    }
  }
  options.set_graph_def_version(flib_def->graph_def_version());
  options.set_allow_cpu_custom_calls(PlatformInfo().platform_id() ==
                                     se::host::kHostPlatformId);

  options.set_use_tuple_arg(false);
  options.set_return_updated_values_for_all_resources(false);
  options.set_resolve_compile_time_constants(true);
  options.set_always_return_tuple(false);
  options.set_is_entry_computation(true);

  for (int i = 0; i < ctx->num_outputs(); ++i) {
    if (ctx->output_memory_type(i) == DEVICE_MEMORY &&
        ctx->op_device_context()) {
      options.add_output_placements("gpu");
    } else {
      options.add_output_placements("cpu");
    }
  }

  std::map<int, Tensor> constant_args;
  for (int i : ConstantsAttr()) {
    constant_args.insert({i, ctx->input(i)});
  }

  std::set<int> fixed_shape_args;
  std::set<int> host_args_set;
  fixed_shape_args.insert(FixedShapesAttr().begin(), FixedShapesAttr().end());
  host_args_set.insert(HostArgsAttr().begin(), HostArgsAttr().end());
  auto status = cache->Compile(std::move(input_ptr), function, constant_args,
                               fixed_shape_args, host_args_set, *variables, ctx,
                               executable, stat, is_mlir, call_info);
  TaoCompInfoCollector::Get().SetCallTimestamp(call_info,
                                               TIME_COMPILE_CALL_END);
  return status;
}

}  //  namespace tao
}  //  namespace tensorflow

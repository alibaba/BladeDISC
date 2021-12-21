#ifndef TAO_TAO_BRIDGE_KERNELS_LAUNCH_BASE_H_
#define TAO_TAO_BRIDGE_KERNELS_LAUNCH_BASE_H_

#include <sstream>

#include "tao_bridge/common.h"
#include "tao_bridge/cuda_utils.h"
#include "tao_bridge/kernels/platform_info.h"
#include "tao_bridge/kernels/profiling.h"
#include "tao_bridge/kernels/tao_compilation_info_collector.h"
#include "tao_bridge/kernels/tao_profiling_guided_compilation.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/stream_executor/platform.h"

namespace tensorflow {
namespace tao {

constexpr static int64 kTickPrintFreq = 1000;

// CompilationMode indicates sync(default) or async compilation mode
enum CompilationMode {
  // Default is to use sync compilation
  kDefault = 0,
  // Call Tao Executable when compilation finish,
  // or else fallback to TF execution.
  kAsync = 1,
};

template <bool>
struct assign_if_not_const;

template <>
struct assign_if_not_const<false> {
  template <typename T, typename V>
  void Run(T&& lhs, V&& rhs) {
    lhs = rhs;
  }
};

template <>
struct assign_if_not_const<true> {
  template <typename T, typename V>
  void Run(T&& lhs, V&& rhs) {}
};

class DoneHelper : public tensorflow::core::RefCounted {
 public:
  DoneHelper(AsyncOpKernel::DoneCallback done) {
    done_ = done;
    TaoCompInfoCollector::Get().SetCallTimestamp(&call_info,
                                                 TIME_TAO_LAUNCH_OP_EXEC_BEGIN);
  }
  ~DoneHelper() override {
    done_();
    TaoCompInfoCollector::Get().SetCallTimestamp(&call_info,
                                                 TIME_TAO_LAUNCH_OP_EXEC_END);
    TaoCompInfoCollector::Get().FlushCallTimestamp(&call_info);
  }

 private:
  AsyncOpKernel::DoneCallback done_;

 public:
  TaoCompileFuncCallInfo call_info;
};

class LaunchBase : public AsyncOpKernel {
 public:
  explicit LaunchBase(OpKernelConstruction* ctx);

  const std::vector<int>& ConstantsAttr() { return constants_; }
  const std::vector<int>& FixedShapesAttr() { return fixed_shapes_; }
  const std::vector<int>& HostArgsAttr() { return host_args_; }
  const std::vector<int>& ResourcesAttr() { return resources_; }
  PlatformInfo& PlatformInfoCtx() { return platform_info_; }
  std::string DeviceUUIDCtx() { return device_uuid_; }

  // return function attr from the given name
  NameAttrList FunctionAttr(const char* const attr);
  // return bool attr value from the given name
  bool BoolAttr(const char* const attr);

  Status DeviceUUIDFromContext(std::string* result_uuid);

  void printInOuts(OpKernelContext* ctx);

  Tensor ToCpu(OpKernelContext* ctx, Tensor t, MemoryType mem_type);

  std::map<int, OptionalTensor> SnapshotResourceVariables(
      OpKernelContext* ctx, absl::Span<const int> variables);

  // Launch Executable helper functions
  Status EnsureFunctionHandle(OpKernelContext* ctx, const NameAttrList& func,
                              FunctionLibraryRuntime::Handle* handle);
  Status PrepareInputsCpu(OpKernelContext* ctx,
                          std::vector<Tensor>* inputs_ptr);
  Status PrepareOutputsCpu(OpKernelContext* ctx, std::vector<Tensor>* outputs);
  Status SyncInputsMem(OpKernelContext* ctx, std::vector<Tensor>* inputs_ptr,
                       bool check_block);
  Status SyncOutputsMem(OpKernelContext* ctx, std::vector<Tensor>* outputs,
                        bool check_block);
  Status RunExecutable(Executable* executable, ExecutableRunOptions& options,
                       TaoCompileFuncCallInfo* call_info);

  template <typename T = GpuTFProfiler>
  Status ExecuteFunction(const NameAttrList& func,
                         FunctionLibraryRuntime::Handle* func_handle,
                         OpKernelContext* ctx, DoneHelper* helper,
                         TFProfiler<T>* profiler = nullptr) {
    TaoCompInfoCollector::Get().SetCallTimestamp(&(helper->call_info),
                                                 TIME_FUNC_RUN_BEGIN);

    FunctionLibraryRuntime* func_lib = ctx->function_library();
    tensorflow::FunctionLibraryRuntime::Options opts;
    // We need this because in the latest tensorflow `opts.step_id` has type
    // `const int64` while in the former version it has type `int64`
    // Comment by muzhuo.yj to further discuss with wenyi:
    //  So is this conditional assignment a little bit tricky and bug-prone?
    //  For "const int64 step_id", if we don't do the assignment, what will
    //  happens?
    // Reply: in higher version it will be initialized by the default
    // constructor of `tensorflow::FunctionLibraryRuntime::Options` and its
    // value is guaranteed to be unique, thus we do not need to set this value.
    assign_if_not_const<std::is_const<decltype(opts.step_id)>::value>().Run(
        opts.step_id, ctx->step_id());
    opts.rendezvous = ctx->rendezvous();
    opts.cancellation_manager = ctx->cancellation_manager();
    opts.step_container = ctx->step_container();
    opts.stats_collector = ctx->stats_collector();
    opts.runner = ctx->runner();
    opts.collective_executor = ctx->collective_executor();

    std::vector<Tensor>* outputs = new std::vector<Tensor>();
    TF_RETURN_IF_ERROR(EnsureFunctionHandle(ctx, func, func_handle));
    CHECK_NE(*func_handle, tensorflow::kInvalidHandle);
    VLOG(2) << "Executing func = " << func.name();

    if (profiler) {
      profiler->Start(ctx);
    }
    std::vector<Tensor> inputs;
    if (ctx->op_device_context() != nullptr) {
      TF_RETURN_IF_ERROR(SyncInputsMem(ctx, &inputs, !profiler));
    } else {
      // For cpu inputs handling
      TF_RETURN_IF_ERROR(PrepareInputsCpu(ctx, &inputs));
    }

    if (profiler) {
      profiler->RecordComputationStart(ctx);
    }
    helper->Ref();  // Increment count for calculating native graph
    func_lib->Run(
        opts, *func_handle, inputs, outputs,
        [this, ctx, outputs, helper, profiler](const tensorflow::Status& s) {
          tensorflow::core::ScopedUnref sc(helper);
          TaoCompInfoCollector::Get().SetCallTimestamp(&(helper->call_info),
                                                       TIME_FUNC_RUN_END);

          VLOG(2) << "Native Segment completed";
          if (!s.ok()) {
            LOG(WARNING) << "TaoLaunch execute function failed: "
                         << s.error_message();
            ctx->SetStatus(s);
            if (profiler) {
              profiler->RecordComputationFinish(ctx);
              profiler->Stop(s);
              delete profiler;
            }
            return;
          }
          CHECK_EQ(static_cast<int>(outputs->size()), ctx->num_outputs());
          if (profiler) {
            profiler->RecordComputationFinish(ctx);
          }
          if (ctx->op_device_context() != nullptr) {
            this->SyncOutputsMem(ctx, outputs, !profiler);
          } else {
            // For cpu outputs handling
            PrepareOutputsCpu(ctx, outputs);
          }
          if (profiler) {
            profiler->Stop(s);
            delete profiler;
          }
          if (GetTaoBridgeOptions()->disc_debug_mode) {
            VLOG(0) << "print ins/outs from fallback";
            printInOuts(ctx);
          }
          delete outputs;
        });
    return Status::OK();
  }
  Status PrepareExecutableRunOptions(OpKernelContext* ctx,
                                     int num_constant_args,
                                     std::map<int, OptionalTensor> variables,
                                     ExecutableRunOptions* options) {
    TF_RET_CHECK(options != nullptr);
    (*options)
        .set_ctx(ctx)
        .set_num_constant_args(num_constant_args)
        .set_variables(variables);
    return Status::OK();
  }

 private:
  // return attr->constants as vector
  std::vector<int> ConstantsVector();
  // return attr->fixed_shapes as vector
  std::vector<int> FixedShapesVector();
  // return attr->host_args as vector
  std::vector<int> HostArgsVector();
  // return attr->resources as vector
  std::vector<int> ResourcesVector();

  OpKernelConstruction* ctx_;
  std::vector<int> constants_;
  std::vector<int> fixed_shapes_;
  std::vector<int> host_args_;
  std::vector<int> resources_;
  PlatformInfo platform_info_;
  std::string device_uuid_;
};

// TaoLaunchTick records the run times of TAO ops
class TaoLaunchTick {
 public:
  TaoLaunchTick(std::string name, int64 print_freq = kTickPrintFreq)
      : name_(name),
        print_freq_(print_freq),
        total_times_(0),
        prev_total_times_(0),
        tao_times_(0),
        prev_tao_times_(0){};
  // increase total run times
  void IncressTotal() { total_times_++; }

  // increase tao run times
  void IncressTao() { tao_times_++; }

  // return true if should print
  bool ShouldPrint() {
    return total_times_ > 0 && total_times_ % print_freq_ == 0;
  };

  std::string Report() {
    std::stringstream ss;
    ss << "#cluster(" << name_ << ") "
       << "All: Tao/Total = " << tao_times_ << "/" << total_times_
       << ", Recent: Tao/Total = " << (tao_times_ - prev_tao_times_) << "/"
       << (total_times_ - prev_total_times_);
    prev_total_times_ = total_times_;
    prev_tao_times_ = tao_times_;
    return ss.str();
  };

 private:
  std::string name_;
  int64 print_freq_;
  int64 total_times_;
  int64 prev_total_times_;
  int64 tao_times_;
  int64 prev_tao_times_;
};

}  //  namespace tao
}  //  namespace tensorflow

#endif  //  TAO_TAO_BRIDGE_KERNELS_LAUNCH_BASE_H_
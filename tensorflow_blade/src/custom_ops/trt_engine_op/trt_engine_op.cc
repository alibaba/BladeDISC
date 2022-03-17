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

#define EIGEN_USE_GPU
#include <mutex>

#include "NvInferVersion.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "src/tensorrt/bridge/tensorrt_common.h"
#include "src/tensorrt/bridge/tensorrt_logger.h"
#include "src/tensorrt/bridge/tensorrt_tf_allocator.h"
#include "src/tensorrt/bridge/tensorrt_tf_resource_mgr.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

using ::tf_blade::trt::TRTBaseAllocator;
using ::tf_blade::trt::TRTCacheResource;
using ::tf_blade::trt::TRTDeviceAllocator;
using ::tf_blade::trt::TrtUniquePtr;

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace {

static constexpr const char* kBladeTrtContainerName = "BLADE-TRT";

}

void NvinferExecutionContextDeleter(nvinfer1::IExecutionContext* ctx) {
  if (ctx != nullptr) {
    ctx->destroy();
  }
}

#define TYPECASE(dt, X, Y)                                    \
  case dt: {                                                  \
    return (void*)X->flat<EnumToDataType<dt>::Type>().data(); \
  }

void* GetTensorAddress(const Tensor* tensor_ptr) {
  auto tensor_type = tensor_ptr->dtype();
  switch (tensor_type) {
    TYPECASE(DT_FLOAT, tensor_ptr, dest_ptr);
    TYPECASE(DT_HALF, tensor_ptr, dest_ptr);
    TYPECASE(DT_INT8, tensor_ptr, dest_ptr);
    TYPECASE(DT_INT32, tensor_ptr, dest_ptr);
    default: {
      LOG(ERROR) << "Unsupported Data type " << DataTypeString(tensor_type);
      return nullptr;
    }
  }
}

class TrtEngineOp : public AsyncOpKernel {
 public:
  explicit TrtEngineOp(OpKernelConstruction* context) : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("input_names", &input_names_));
    OP_REQUIRES_OK(context, context->GetAttr("Tin", &input_types_));
    CHECK_EQ(input_names_.size(), input_types_.size());
    OP_REQUIRES_OK(context, context->GetAttr("output_names", &output_names_));
    OP_REQUIRES_OK(context, context->GetAttr("Tout", &output_types_));
    CHECK_EQ(output_names_.size(), output_types_.size());
    OP_REQUIRES_OK(context, context->GetAttr("engine_bytes", &engine_bytes_));
    VLOG(2) << "engine_bytes size: " << engine_bytes_.size();

    // Setup fallback function
    OP_REQUIRES_OK(
        context, context->GetAttr("fallback_function", &tf_fallback_function_));
    FunctionLibraryRuntime* func_lib = context->function_library();
    CHECK(func_lib != nullptr) << "Context function library is null";
    FunctionLibraryRuntime::InstantiateOptions inst_ops;
    inst_ops.target = context->device()->name();
    auto status =
        func_lib->Instantiate(tf_fallback_function_.name(),
                              AttrSlice(&(tf_fallback_function_.attr())),
                              inst_ops, &tf_fallback_func_handle_);

    CHECK(status.ok()) << "Instantiating native function "
                       << tf_fallback_function_.name()
                       << " failed. Because of: " << status.error_message();
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    std::call_once(initialized_, [&]() {
      auto status = PrepareTRTExecution(context);
      OP_REQUIRES_ASYNC(
          context, status.ok(),
          errors::Internal("Failed to prepare tensorrt execution due to: ",
                           status.error_message()),
          done);
    });
    if (ValidInputShape(context)) {
      OP_REQUIRES_OK_ASYNC(context, ExecuteTrtEngine(context, done), done);
    } else {
      OP_REQUIRES_OK_ASYNC(context, ExecuteFallbackFunction(context, done),
                           done);
    }
  }

 private:
  Status PrepareTRTExecution(OpKernelContext* context) {
    // Get TRT resource.
    TRTCacheResource* cache_res = nullptr;
    TF_RETURN_IF_ERROR(GetTRTCacheResource(context, &cache_res));
    core::ScopedUnref unref_cache_res(cache_res);
    alloc_ = cache_res->allocator_.get();
    if (!alloc_) {
      return errors::Internal("Failed to get valid allcator!");
    }

    engine_.reset(CreateInferRuntime(engine_bytes_, alloc_));
    if (!engine_) {
      return errors::Internal("Failed to deserialize trt engine!");
    }
    std::string().swap(engine_bytes_);
    // According to
    // https://forums.developer.nvidia.com/t/nvinfer1-createexecutioncontextwithoutdevicememory-returns-nullptr/111878
    // createExecutionContextWithoutDeviceMemory comes with a certain bug
    // which is fixed in version after trt-7.2
#if NV_TENSORRT_MAJOR > 7
    exec_ctx_.reset(engine_->createExecutionContextWithoutDeviceMemory(),
                    NvinferExecutionContextDeleter);
#else
    exec_ctx_.reset(engine_->createExecutionContext(),
                    NvinferExecutionContextDeleter);
#endif
    if (!exec_ctx_) {
      return errors::Internal("Failed to create execution context!");
    }
#if NV_TENSORRT_MAJOR > 7
    // Allocate device memory for the TensorRT engine execution. The device
    // memory will be released when context_device_memory goes out of scope.
    ::tf_blade::trt::ContextDeviceMemory context_device_memory;
    TF_RETURN_IF_ERROR(
        context_device_memory.AllocateDeviceMemory(exec_ctx_.get(), alloc_));
#endif
    for (int i = 0; i < context->num_inputs(); i++) {
      auto min_dims = engine_->getProfileDimensions(
          i, 0, nvinfer1::OptProfileSelector::kMIN);
      auto max_dims = engine_->getProfileDimensions(
          i, 0, nvinfer1::OptProfileSelector::kMAX);
      CHECK(min_dims.nbDims == max_dims.nbDims)
          << "Max and Min dimension in TensorRT profiles should have the same "
             "rank!";
      for (int j = 0; j < min_dims.nbDims; j++) {
        if (min_dims.d[j] != max_dims.d[j]) {
          is_dynamic_engine_ = true;
          break;
        }
      }
      if (is_dynamic_engine_) {
        int bind_index = engine_->getBindingIndex(input_names_[i].c_str());
        if (bind_index >= 0) {
          exec_ctx_->setBindingDimensions(bind_index, min_dims);
        }
      }
    }
    return Status::OK();
  }

  Status GetTRTCacheResource(OpKernelContext* ctx,
                             TRTCacheResource** cache_res) {
    // Canonicalize the op name by removing the scopes if any. This is mainly
    // because in TFv2, the function graph can be instantiated in various ways
    // and it'll insert scope names to the name of the TRTEngineOps, which will
    // result in many different engine caches if we use the instantiated op name
    // directly, but we still want all of them share the same cache (if they
    // were representing the same subgraph).
    absl::string_view resource_name = name();
    size_t last_slash = resource_name.find_last_of('/');
    if (last_slash != absl::string_view::npos) {
      resource_name.remove_prefix(last_slash + 1);
    }

    // Get trt cache.
    // NOTE(lanbo.llb): OpKernelConstruction does not have access to
    // resource_maneger in tf 1.15, thus we have to get ResourceManager* where
    // OpKernelContext is available as we do here in ComputeAsync, not in Op's
    // Construction
    return ctx->resource_manager()->LookupOrCreate(
        std::string(kBladeTrtContainerName), std::string(resource_name),
        cache_res, {[ctx](TRTCacheResource** cr) -> Status {
          *cr = new TRTCacheResource(ctx);
          return Status::OK();
        }});
  }

  nvinfer1::ICudaEngine* CreateInferRuntime(const std::string& engine_data,
                                            TRTBaseAllocator* allocator) {
    auto& logger = ::tf_blade::trt::GetTensorrtLogger();
    // The initLibNvInferPlugins would be called once
    bool ret = ::tf_blade::trt::InitializeTrtPlugins(&logger);
    if (ret) {
      const TrtUniquePtr<nvinfer1::IRuntime> infer{
          nvinfer1::createInferRuntime(logger)};
      infer->setGpuAllocator(allocator);
      auto* engine_ptr = infer->deserializeCudaEngine(
          engine_data.data(), engine_data.size(), nullptr);
      return engine_ptr;
    } else {
      return nullptr;
    }
  }

  bool ValidInputShape(OpKernelContext* context) {
    for (int i = 0; i < context->num_inputs(); i++) {
      auto min_dims = engine_->getProfileDimensions(
          i, 0, nvinfer1::OptProfileSelector::kMIN);
      auto max_dims = engine_->getProfileDimensions(
          i, 0, nvinfer1::OptProfileSelector::kMAX);
      auto input_shape = context->input(i).shape();
      // Rank mismatch
      if (min_dims.nbDims != input_shape.dims()) {
        return false;
      }
      for (int j = 0; j < input_shape.dims(); j++) {
        auto dim = input_shape.dim_size(j);
        if (dim < min_dims.d[j] || dim > max_dims.d[j]) {
          // Shape mismatch
          return false;
        }
      }
    }
    return true;
  }

  void BindingInputs(OpKernelContext* context,
                     std::vector<void*>& binding_buffers, DoneCallback done) {
    for (int i = 0; i < context->num_inputs(); i++) {
      auto input_tensor = context->input(i);
      int bind_index = engine_->getBindingIndex(input_names_[i].c_str());
      if (is_dynamic_engine_) {
        auto dims = engine_->getBindingDimensions(bind_index);
        auto input_shape = input_tensor.shape();
        for (int j = 0; j < input_shape.dims(); j++) {
          dims.d[j] = input_shape.dim_size(j);
        }
        exec_ctx_->setBindingDimensions(bind_index, dims);
      }
      binding_buffers[bind_index] = GetTensorAddress(&input_tensor);
      OP_REQUIRES_ASYNC(context, binding_buffers[bind_index],
                        errors::InvalidArgument(
                            "Unsupported data type encountered in input ", i),
                        done);
    }
  }

  void BindingOutputs(OpKernelContext* context,
                      std::vector<void*>& binding_buffers, DoneCallback done) {
    for (int i = 0; i < context->num_outputs(); i++) {
      int bind_index = engine_->getBindingIndex(output_names_[i].c_str());
      const auto& dims = exec_ctx_->getBindingDimensions(bind_index);
      std::vector<int64> shape(dims.nbDims);
      for (int j = 0; j < dims.nbDims; ++j) {
        shape[j] = static_cast<int64>(dims.d[j]);
      }
      TensorShape tensor_shape(shape);
      Tensor* output = nullptr;
      context->allocate_output(i, tensor_shape, &output);
      binding_buffers[bind_index] = GetTensorAddress(output);
      OP_REQUIRES_ASYNC(context, binding_buffers[bind_index],
                        errors::InvalidArgument(
                            "Unsupported data type encountered in output ", i),
                        done);
    }
  }

  Status ExecuteTrtEngine(OpKernelContext* context, DoneCallback done) {
#if NV_TENSORRT_MAJOR > 7
    ::tf_blade::trt::ContextDeviceMemory context_device_memory;
    auto allocate_ret =
        context_device_memory.AllocateDeviceMemory(exec_ctx_.get(), alloc_);
    if (!allocate_ret.ok()) {
#if TF_MAJOR == 2
      LOG_FIRST_N(WARNING, 5)
          << "Failed to allocate device memory for execution context, will "
             "fallback to tf execution!";
#endif
      return ExecuteFallbackFunction(context, done);
    }
#endif
    // get tensorflow default cuda stream
    const cudaStream_t& stream = context->eigen_device<GPUDevice>().stream();
    int num_bindings = engine_->getNbBindings();
    VLOG(2) << "num_bindings: " << num_bindings;
    std::vector<void*> binding_buffers(num_bindings);
    bool ret;
    {
      // lock guard for the use of NvinferExecutionContext which is not
      // thread-safe
      std::lock_guard<std::mutex> guard(lock_);
      BindingInputs(context, binding_buffers, done);
      BindingOutputs(context, binding_buffers, done);
      ret = exec_ctx_->enqueueV2(&binding_buffers[0], stream, nullptr);
      if (!ret) {
#if TF_MAJOR == 2
        LOG_FIRST_N(WARNING, 5) << "Failed to execute tensorrt kernels, will "
                                   "fallback to tf execution!";
#endif
        return ExecuteFallbackFunction(context, done);
      }
    }

    done();
    return Status::OK();
  }

  Status ExecuteFallbackFunction(OpKernelContext* context, DoneCallback done) {
    FunctionLibraryRuntime* func_lib = context->function_library();
    CHECK(func_lib != nullptr) << "Context function library is null";

    std::vector<Tensor> inputs;
    for (int i = 0; i < context->num_inputs(); ++i) {
      auto& input_tensor = context->input(i);
      inputs.push_back(input_tensor);
    }

    tensorflow::FunctionLibraryRuntime::Options opts;
    opts.rendezvous = context->rendezvous();
    opts.cancellation_manager = context->cancellation_manager();
    opts.step_container = context->step_container();
    opts.stats_collector = context->stats_collector();
    opts.runner = context->runner();
    opts.collective_executor = context->collective_executor();

    std::vector<Tensor>* outputs = new std::vector<Tensor>();
    VLOG(2) << "Executing func = " << tf_fallback_function_.name();

    func_lib->Run(opts, tf_fallback_func_handle_, inputs, outputs,
                  [context, outputs, done](const Status& s) {
                    for (int i = 0; i < context->num_outputs(); ++i) {
                      auto& output_tensor = outputs->at(i);
                      context->set_output(i, outputs->at(i));
                    }
                    done();
                  });

    return Status::OK();
  }

  std::vector<std::string> input_names_;
  std::vector<tensorflow::DataType> input_types_;
  std::vector<std::string> output_names_;
  std::vector<tensorflow::DataType> output_types_;
  std::string engine_bytes_;
  TrtUniquePtr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IExecutionContext> exec_ctx_;
  bool is_dynamic_engine_;
  mutable std::mutex lock_;
  TRTBaseAllocator* alloc_;
  std::once_flag initialized_;
  FunctionLibraryRuntime::Handle tf_fallback_func_handle_{kInvalidHandle};
  // BladeTrtEngine Op hold a func attr, make the related function in
  // FunctionLibrary is reachable which cannot be pruned by grappler. Then trt
  // cannot work with amp pass together
  NameAttrList tf_fallback_function_;
};

REGISTER_OP("BladeTrtEngine")
    .Input("input: Tin")
    .Output("output: Tout")
    .Attr("Tin: list(type) >= 1")
    .Attr("Tout: list(type) >= 1")
    .Attr("input_names: list(string) >= 1")
    .Attr("input_shapes: list(shape)")
    .Attr("output_names: list(string) >= 1")
    .Attr("engine_bytes: string")
    .Attr("fallback_function: func")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_KERNEL_BUILDER(Name("BladeTrtEngine").Device(DEVICE_GPU), TrtEngineOp);

}  // namespace tensorflow

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Adopted from tensorflow/core/kernels/tensor_array_ops.cc.
// This is used to register a set of TensorArray-related kernels with type int32
// on GPU, which will cheat the Placer to put these ops on GPU for compilation
// of control flow ops. There is no implementation provided, as we will
// replace these ops on CPU for normal tf execution.

#define EIGEN_USE_THREADS

#include <limits>
#include <numeric>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
// #include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/dynamic_annotations.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/ptr_util.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace tensorflow {

#define TAO_REGISTER_GPU(m) TAO_REGISTER_GPU_UNIQ_HELPER(__COUNTER__, m)
#define TAO_REGISTER_GPU_UNIQ_HELPER(ctr, m) TAO_REGISTER_GPU_UNIQ(ctr, m)
#define TAO_REGISTER_GPU_UNIQ(ctr, m)                                          \
  namespace {                                                                  \
  struct _tao_register_##ctr {                                                 \
    _tao_register_##ctr() {                                                    \
      bool enable_tao;                                                         \
      auto status = tensorflow::ReadBoolFromEnvVar("BRIDGE_ENABLE_TAO", false, \
                                                   &enable_tao);               \
      bool enable_control_flow;                                                \
      status = tensorflow::ReadBoolFromEnvVar("TAO_ENABLE_CONTROL_FLOW",       \
                                              false, &enable_control_flow);    \
      if (enable_tao && enable_control_flow) {                                 \
        TF_CALL_int32(m);                                                      \
      }                                                                        \
    }                                                                          \
  };                                                                           \
  static _tao_register_##ctr _tao_register_##ctr##_object;                     \
  }

// A per-run local tensor array. The tensor array uses a "per-step" resource
// manager which ensures that correct garbage collection on error or
// successful completion.
class TensorArrayOpBogon : public OpKernel {
 public:
  explicit TensorArrayOpBogon(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    LOG(FATAL) << "TensorArrayOp device='GPU' dtype=DT_INT32 Not Implemented";
    return;
  }
};

#define REGISTER_GPU(type)                                   \
  REGISTER_KERNEL_BUILDER(Name("TensorArrayV3")              \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<type>("dtype") \
                              .HostMemory("size")            \
                              .HostMemory("handle"),         \
                          TensorArrayOpBogon);

TAO_REGISTER_GPU(REGISTER_GPU);

#undef REGISTER_GPU

template <typename Device, typename T>
class TensorArrayWriteOpBogon : public OpKernel {
 public:
  explicit TensorArrayWriteOpBogon(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    LOG(FATAL)
        << "TensorArrayWriteOp device='GPU' dtype=DT_INT32 Not Implemented";
    return;
  }
};

#define REGISTER_GPU(type)                               \
  REGISTER_KERNEL_BUILDER(Name("TensorArrayWriteV3")     \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("handle")      \
                              .HostMemory("index"),      \
                          TensorArrayWriteOpBogon<GPUDevice, type>);

TAO_REGISTER_GPU(REGISTER_GPU);

#undef REGISTER_GPU

template <typename Device, typename T>
class TensorArrayReadOpBogon : public OpKernel {
 public:
  explicit TensorArrayReadOpBogon(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    LOG(FATAL)
        << "TensorArrayReadOp device='GPU' dtype=DT_INT32 Not Implemented";
    return;
  }
};

#define REGISTER_GPU(type)                                   \
  REGISTER_KERNEL_BUILDER(Name("TensorArrayReadV3")          \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<type>("dtype") \
                              .HostMemory("handle")          \
                              .HostMemory("index"),          \
                          TensorArrayReadOpBogon<GPUDevice, type>);

TAO_REGISTER_GPU(REGISTER_GPU);

#undef REGISTER_GPU

template <typename Device, typename T, bool LEGACY_UNPACK>
class TensorArrayUnpackOrScatterOpBogon : public OpKernel {
 public:
  explicit TensorArrayUnpackOrScatterOpBogon(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    LOG(FATAL) << "TensorArrayUnpackOrScatterOp device='GPU' dtype=DT_INT32 "
                  "Not Implemented";
    return;
  }
};

#define REGISTER_GPU(type)                                          \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("TensorArrayUnpack")                                     \
          .Device(DEVICE_GPU)                                       \
          .TypeConstraint<type>("T")                                \
          .HostMemory("handle"),                                    \
      TensorArrayUnpackOrScatterOpBogon<GPUDevice, type,            \
                                        true /* LEGACY_UNPACK */>); \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("TensorArrayScatterV3")                                  \
          .Device(DEVICE_GPU)                                       \
          .TypeConstraint<type>("T")                                \
          .HostMemory("indices")                                    \
          .HostMemory("handle"),                                    \
      TensorArrayUnpackOrScatterOpBogon<GPUDevice, type,            \
                                        false /* LEGACY_UNPACK */>);

TAO_REGISTER_GPU(REGISTER_GPU);

#undef REGISTER_GPU

}  // namespace tensorflow

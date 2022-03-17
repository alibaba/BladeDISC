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

#ifdef GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
struct ElementWiseLauncher {
  static void Run(cudaStream_t stream, const int hiddenSize,
                  const int miniBatch, const T* tmp_h, const T* tmp_i,
                  const T* bias, const T* forget_bias, T* h_out, const T* c_in,
                  T* c_out);
};

template <typename T>
class FusedLSTMElementWiseOp : public OpKernel {
 public:
  explicit FusedLSTMElementWiseOp(OpKernelConstruction* c) : OpKernel(c) {}
  ~FusedLSTMElementWiseOp() {}
  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<T>();
    const Tensor& c_in_tensor = context->input(1);
    auto c_in = c_in_tensor.flat<T>();
    const Tensor& b_in_tensor = context->input(2);
    auto b_in = b_in_tensor.flat<T>();
    const Tensor& forget_bias_tensor = context->input(3);
    auto forget_bias = forget_bias_tensor.flat<T>();

    int miniBatch = c_in_tensor.shape().dim_size(0);
    int hiddenSize = c_in_tensor.shape().dim_size(1);

    Tensor* c_out_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, c_in_tensor.shape(),
                                                     &c_out_tensor));
    auto c_out = c_out_tensor->flat<T>();
    Tensor* h_out_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, c_in_tensor.shape(),
                                                     &h_out_tensor));
    auto h_out = h_out_tensor->flat<T>();

    // If there is nothing to compute, return.
    if (c_out_tensor->shape().num_elements() == 0 ||
        h_out_tensor->shape().num_elements() == 0) {
      return;
    }

    const GPUDevice& d = context->eigen_device<GPUDevice>();
    ElementWiseLauncher<T>().Run(d.stream(), hiddenSize, miniBatch, NULL,
                                 input.data(), b_in.data(), forget_bias.data(),
                                 h_out.data(), c_in.data(), c_out.data());
  }
};

#define REGISTER_GPU(T)                                     \
  REGISTER_KERNEL_BUILDER(Name("BladeFusedLSTMElementWise") \
                              .Device(DEVICE_GPU)           \
                              .TypeConstraint<T>("T"),      \
                          FusedLSTMElementWiseOp<T>);
REGISTER_GPU(float);
REGISTER_GPU(Eigen::half);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA

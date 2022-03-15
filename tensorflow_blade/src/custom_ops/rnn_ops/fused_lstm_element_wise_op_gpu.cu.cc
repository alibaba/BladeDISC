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

#include "fused_lstm_element_wise_op_kernels.cu.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

template <typename T>
struct ElementWiseLauncher {
  static void Run(cudaStream_t stream, const int hiddenSize,
                  const int miniBatch, const T* tmp_h, const T* tmp_i,
                  const T* bias, const T* forget_bias, T* h_out, const T* c_in,
                  T* c_out) {
    const int bsize = 256;
    dim3 gridDim = (hiddenSize * miniBatch + bsize - 1) / bsize;
    ElementWise<T><<<gridDim, bsize, 0, stream>>>(hiddenSize, miniBatch, tmp_h,
                                                  tmp_i, bias, forget_bias,
                                                  h_out, c_in, c_out);
  }
};

template struct ElementWiseLauncher<float>;
template struct ElementWiseLauncher<Eigen::half>;

}  // namespace tensorflow

#endif  // GOOGLE_CUDA

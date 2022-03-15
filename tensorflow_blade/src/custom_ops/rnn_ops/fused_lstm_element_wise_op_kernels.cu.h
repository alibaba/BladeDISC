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

#ifndef RNN_OPS_FUSED_LSTM_ELEMENT_WISE_OP_KERNELS_H_
#define RNN_OPS_FUSED_LSTM_ELEMENT_WISE_OP_KERNELS_H_

#ifdef GOOGLE_CUDA

#define EIGEN_USE_GPU

namespace tensorflow {

template <typename T>
__inline__ __device__ T sigmoid(T in) {
  return (T)1.f / ((T)1.f + exp(-in));
}

template <typename T>
__global__ void ElementWise(const int hiddenSize, const int miniBatch,
                            const T* tmp_h, const T* tmp_i, const T* bias,
                            const T* forget_bias, T* h_out, const T* c_in,
                            T* c_out) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int numElements = miniBatch * hiddenSize;
  if (index >= numElements) return;
  int batch = index / hiddenSize;
  int gateIndex = (index % hiddenSize) + 4 * batch * hiddenSize;

  T g[4];
  for (int i = 0; i < 4; ++i) {
    if (tmp_h == NULL)
      g[i] = (tmp_i[i * hiddenSize + gateIndex]);
    else
      g[i] = (tmp_i[i * hiddenSize + gateIndex]) +
             (tmp_h[i * hiddenSize + gateIndex]);
    if (bias != NULL) g[i] += (bias[i * hiddenSize + index % hiddenSize]);
  }
  T in_gate = sigmoid<T>(g[0]);
  T c_update = tanh(g[1]);
  T forget_gate = sigmoid<T>(forget_bias[0] + g[2]);
  T out_gate = sigmoid<T>(g[3]);

  T val;
  if (c_in == NULL)
    val = in_gate * c_update;
  else
    val = forget_gate * (c_in[index]) + in_gate * c_update;
  if (c_out != NULL) c_out[index] = (val);
  if (h_out != NULL) h_out[index] = out_gate * tanh(val);
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA

#endif  // RNN_OPS_FUSED_LSTM_ELEMENT_WISE_OP_KERNELS_H_

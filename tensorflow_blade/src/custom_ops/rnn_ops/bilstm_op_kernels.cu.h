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

namespace tensorflow {

__inline__ __device__ float sigmoid(float in) {
  return 1.f / (1.f + expf(-in));
}

__inline__ __device__ float dsigmoid(float in) {
  return sigmoid(in) * (1.f - sigmoid(in));
}

__inline__ __device__ float dtanh(float in) {
  return 1.f - (tanhf(in) * tanhf(in));
}

template <typename T, int B, int BX, int BY>
__global__ void BiLstmCell(const int hidden, const int batch, const T* tmp_i_fw,
                           const T* tmp_i_bw, const T* bias_fw, T* h_out_fw,
                           T* h_out_bw, const T* c_in_fw, T* c_out_fw,
                           const T* wh_fw, const T* h_in_fw, const T* wh_bw,
                           const T* h_in_bw) {
  const T* tmp_i;
  const T* bias;
  T* h_out;
  const T* c_in;
  T* c_out;
  const T* wh;
  const T* h_in;

  if (blockIdx.y == 0) {
    tmp_i = tmp_i_fw;
    h_out = h_out_fw;
    wh = wh_fw;
    h_in = h_in_fw;
    bias = bias_fw;
    c_in = c_in_fw;
    c_out = c_out_fw;
  } else {
    tmp_i = tmp_i_bw;
    h_out = h_out_bw;
    wh = wh_bw;
    h_in = h_in_bw;
    bias = bias_fw + hidden * 8;
    c_in = c_in_fw + batch * hidden;
    c_out = c_out_fw + batch * hidden;
  }

  float rg[B][4] = {0.f};
  float rh[B];
  __shared__ float sg[B][4][BX];
  int tx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tx >= hidden) {
    return;
  }
  for (int y = threadIdx.y; y < 4 * B; y += BY) {
    sg[y / 4][y % 4][threadIdx.x] =
        tmp_i[tx + (y % 4) * hidden + (y / 4) * 4 * hidden];
  }
  __syncthreads();
  for (int k = threadIdx.y; k < hidden; k += BY) {
    for (int y = 0; y < B; ++y) {
      rh[y] = h_in[k + y * hidden];
    }
    for (int x = 0; x < 4; ++x) {
      float rw = wh[tx + x * hidden + k * 4 * hidden];
      for (int y = 0; y < B; ++y) {
        rg[y][x] += rh[y] * rw;
      }
    }
  }
  for (int y = 0; y < B; ++y) {
    for (int x = 0; x < 4; ++x) {
      atomicAdd(&sg[y][x][threadIdx.x], rg[y][x]);
    }
  }
  __syncthreads();
  // if (threadIdx.y < B) {
  //   int y = threadIdx.y;
  for (int idy = threadIdx.y; idy < B; idy += BY) {
    int y = idy;
    int index = tx + y * hidden;
    for (int x = 0; x < 4; ++x) {
      rg[0][x] = sg[y][x][threadIdx.x];
      rg[0][x] += bias[x * hidden + tx] + bias[(4 + x) * hidden + tx];
    }
    float in_gate = sigmoid(rg[0][0]);
    float c_update = tanhf(rg[0][2]);
    float forget_gate = sigmoid(rg[0][1] + 0.f);
    float out_gate = sigmoid(rg[0][3]);
    float val;
    if (c_in == NULL) {
      val = in_gate * c_update;
    } else {
      val = forget_gate * (c_in[index]) + in_gate * c_update;
    }
    if (c_out != NULL) c_out[index] = (val);
    if (h_out != NULL) h_out[index] = out_gate * tanhf(val);
  }
}

// template <typename T, int B, int BX, int BY>
// __global__ void LstmCell(const int hidden, const int batch, const T* tmp_i,
//                          const T* bias, T* h_out, const T* c_in, T* c_out,
//                          const T* wh, const T* h_in) {
//   float rg[B][4] = {0.f};
//   float rh[B];
//   __shared__ float sg[B][4][BX];
//   int tx = threadIdx.x + blockIdx.x * blockDim.x;
//   if (tx >= hidden) {
//     return;
//   }
//   if (threadIdx.y < 4 * B) {
//     sg[threadIdx.y / 4][threadIdx.y % 4][threadIdx.x] =
//         tmp_i[tx + (threadIdx.y % 4) * hidden + (threadIdx.y / 4) * 4 *
//         hidden];
//   }
//   __syncthreads();
//   for (int k = threadIdx.y; k < hidden; k += BY) {
//     for (int y = 0; y < B; ++y) {
//       rh[y] = h_in[k + y * hidden];
//     }
//     for (int x = 0; x < 4; ++x) {
//       float rw = wh[tx + x * hidden + k * 4 * hidden];
//       for (int y = 0; y < B; ++y) {
//         rg[y][x] += rh[y] * rw;
//       }
//     }
//   }
//   for (int y = 0; y < B; ++y) {
//     for (int x = 0; x < 4; ++x) {
//       atomicAdd(&sg[y][x][threadIdx.x], rg[y][x]);
//     }
//   }
//   __syncthreads();
//   if (threadIdx.y < B) {
//     int y = threadIdx.y;
//     int index = tx + y * hidden;
//     for (int x = 0; x < 4; ++x) {
//       rg[0][x] = sg[y][x][threadIdx.x];
//       rg[0][x] += bias[x * hidden + tx] + bias[(4 + x) * hidden + tx];
//     }
//     float in_gate = sigmoid(rg[0][0]);
//     float c_update = tanhf(rg[0][2]);
//     float forget_gate = sigmoid(rg[0][1] + 0.f);
//     float out_gate = sigmoid(rg[0][3]);
//     float val;
//     if (c_in == NULL) {
//       val = in_gate * c_update;
//     } else {
//       val = forget_gate * (c_in[index]) + in_gate * c_update;
//     }
//     if (c_out != NULL) c_out[index] = (val);
//     if (h_out != NULL) h_out[index] = out_gate * tanhf(val);
//   }
// }

template <typename T>
__global__ void YConcat(const int hidden, const float* oy_fw,
                        const float* oy_bw, float* oy_concat) {
  const float* i_base;
  float* o_base;
  if (blockIdx.y == 0) {
    i_base = oy_fw;
    o_base = oy_concat + blockIdx.x * 2 * hidden;
  } else {
    i_base = oy_bw;
    o_base = oy_concat + blockIdx.x * 2 * hidden + hidden;
  }
  i_base += blockIdx.x * hidden;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    o_base[i] = i_base[i];
  }
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA

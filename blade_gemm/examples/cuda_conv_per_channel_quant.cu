#include <assert.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>

#include "bladnn/bladnn.h"

#define checkCudaErrors(func)                                                \
  {                                                                          \
    cudaError_t e = (func);                                                  \
    if (e != cudaSuccess)                                                    \
      printf("%s %d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
  }

static inline int getFwdConvDilatedFilterDim(int filterDim, int dilation) {
  return ((filterDim - 1) * dilation) + 1;
}

static inline int getFwdConvPaddedImageDim(int tensorDim, int pad) {
  return tensorDim + (2 * pad);
}

static inline int getFwdConvOutputDim(int tensorDim, int pad, int filterDim,
                                      int stride, int dilation) {
  int p = (getFwdConvPaddedImageDim(tensorDim, pad) -
           getFwdConvDilatedFilterDim(filterDim, dilation)) /
              stride +
          1;
  return (p);
}

// n, h, w, c, k, r, s, stride_h, stride_w
int test_case[9] = {1, 28, 28, 512, 128, 1, 1, 1, 1};

void run_conv2d(int N, int H, int W, int C, int K, int R, int S, int stride_h,
                int stride_w, int pad_h = 1, int pad_w = 1, int dilation_h = 1,
                int dilation_w = 1, int groups = 1) {
  cudaStream_t s;
  cudaStreamCreate(&s);

  bladnn::Context ctx;
  ctx.stream = s;

  bladnn::ConvKind kind = bladnn::ConvKind::kFprop;
  bladnn::Layout data_layout = bladnn::Layout::kNHWC;
  bladnn::Layout kernel_layout = bladnn::Layout::kNHWC;

  bladnn::Dtype in_dtype = bladnn::Dtype::kS8;
  bladnn::Dtype out_dtype = bladnn::Dtype::kS8;

  int P = getFwdConvOutputDim(H, pad_w, R, stride_h, dilation_h);
  int Q = P;

  size_t mem_size_A = N * H * W * C * sizeof(int8_t);
  size_t mem_size_B = K * R * S * C * sizeof(int8_t);
  size_t mem_size_C = N * P * Q * K * sizeof(int8_t);
  size_t mem_size_scale_bias = K * sizeof(float);

  int8_t *ptr_A, *h_A, *ptr_B, *h_B, *ptr_C, *h_C, *h_C_ref;
  float *scale, *h_scale, *bias, *h_bias;

  h_A = (int8_t*)malloc(mem_size_A);
  h_B = (int8_t*)malloc(mem_size_B);
  h_C = (int8_t*)malloc(mem_size_C);
  h_C_ref = (int8_t*)malloc(mem_size_C);
  h_scale = (float*)malloc(mem_size_scale_bias);
  h_bias = (float*)malloc(mem_size_scale_bias);

  /***********Host array initialization***********/
  for (int i = 0; i < N * H * W * C; i += 1) {
    h_A[i] = rand() % 5;
  }
  for (int i = 0; i < K * R * S * C; i += 1) {
    h_B[i] = rand() % 5;
  }
  for (int i = 0; i < N * P * Q * K; i += 1) {
    h_C[i] = 0;
  }
  for (int i = 0; i < K; i += 1) {
    h_scale[i] = rand() % 5;
  }
  for (int i = 0; i < K; i += 1) {
    h_bias[i] = 0;
  }

  float alpha = 1.0f;
  float beta = 0.0f;
  /***********Host array initialization***********/

  checkCudaErrors(cudaMalloc(&ptr_A, mem_size_A));
  checkCudaErrors(cudaMalloc(&ptr_B, mem_size_B));
  checkCudaErrors(cudaMalloc(&ptr_C, mem_size_C));
  checkCudaErrors(cudaMalloc(&scale, mem_size_scale_bias));
  checkCudaErrors(cudaMalloc(&bias, mem_size_scale_bias));

  checkCudaErrors(cudaMemcpy(ptr_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(ptr_B, h_B, mem_size_B, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(ptr_C, h_C, mem_size_C, cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(scale, h_scale, mem_size_scale_bias, cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(bias, h_bias, mem_size_scale_bias, cudaMemcpyHostToDevice));

  bool status = conv2d(&ctx, in_dtype, out_dtype, kind, data_layout,
                       kernel_layout, N, H, W, C, K, R, S, P, Q, pad_h, pad_w,
                       stride_h, stride_w, dilation_h, dilation_w, groups,
                       ptr_A, ptr_B, ptr_C, &alpha, &beta, scale, bias);

  assert(status == true);

  checkCudaErrors(cudaMemcpy(h_C, ptr_C, mem_size_C, cudaMemcpyDeviceToHost));

  // Host Computation
  for (int n = 0; n < N; ++n) {
    for (int p = 0; p < P; ++p) {
      for (int q = 0; q < Q; ++q) {
        for (int k = 0; k < K; ++k) {
          int32_t acc = 0;

          for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
              for (int c = 0; c < C; ++c) {
                int filter_r = r;
                int filter_s = s;

                int h = p * stride_h - pad_h + filter_r * dilation_h;
                int w = q * stride_w - pad_w + filter_s * dilation_w;

                if (h >= 0 && h < H && w >= 0 && w < W) {
                  int8_t a = h_A[n * H * W * C + h * W * C + w * C + c];
                  int8_t b = h_B[k * R * S * C + r * S * C + s * C + c];

                  acc += int32_t(a) * int32_t(b);
                }
              }
            }
          }

          h_C_ref[n * P * Q * K + p * Q * K + q * K + k] =
              int8_t(std::max(float(0), float(acc * h_scale[k] + h_bias[k])));
        }
      }
    }
  }
  bool correct = true;
  double eps = 1.e-6;

  for (int i = 0; i < N * P * Q * K; i++) {
    double abs_err = fabs(h_C[i] - h_C_ref[i]);
    double dot_length = N * P * Q;
    double abs_val = fabs(h_C[i]);
    double rel_err = abs_err / abs_val / dot_length;
    if (rel_err > eps) {
      printf("Error! Matrix[%05d]=%d, ref=%d error term is > %E\n", i,
             h_C_ref[i], h_C[i], eps);
      correct = false;
      break;
    }
  }

  if (correct)
    printf("Result = Pass\n");
  else
    printf("Result = Fail\n");

  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_ref);
  free(h_scale);
  free(h_bias);

  cudaFree(ptr_A);
  cudaFree(ptr_B);
  cudaFree(ptr_C);
  cudaFree(scale);
  cudaFree(bias);
}

int main() {
  run_conv2d(test_case[0], test_case[1], test_case[2], test_case[3],
             test_case[4], test_case[5], test_case[6], test_case[7],
             test_case[8]);
  return 0;
}
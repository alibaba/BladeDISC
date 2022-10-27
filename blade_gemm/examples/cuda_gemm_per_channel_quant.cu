#include <assert.h>
#include <bladnn/bladnn.h>
#include <cuda.h>

#include <algorithm>
#include <cstdio>

#define checkCudaErrors(func)                                                \
  {                                                                          \
    cudaError_t e = (func);                                                  \
    if (e != cudaSuccess)                                                    \
      printf("%s %d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
  }

int main() {
  cudaStream_t s;
  cudaStreamCreate(&s);

  bladnn::Context ctx;
  ctx.stream = s;

  int M = 512;
  int K = 512;
  int N = 512;

  size_t mem_size_A = M * K * sizeof(int8_t);
  size_t mem_size_B = K * N * sizeof(int8_t);
  size_t mem_size_C = M * N * sizeof(int8_t);
  size_t mem_size_scale_bias = N * sizeof(float);

  bladnn::Dtype a_dtype = bladnn::Dtype::kS8;
  bool a_transpose = 0;
  int8_t *ptr_A, *h_A;
  int a_dim0 = M, a_dim1 = K;

  bladnn::Dtype b_dtype = bladnn::Dtype::kS8;
  bool b_transpose = 1;
  int8_t *ptr_B, *h_B;
  int b_dim0 = N, b_dim1 = K;

  bladnn::Dtype c_dtype = bladnn::Dtype::kS8;
  int8_t *ptr_C, *h_C, *h_C_ref;
  int c_dim0 = M, c_dim1 = N;

  float *scale, *h_scale;
  float *bias, *h_bias;

  float alpha = 1.0, beta = 0.0;

  int batch_count = 1;

  h_A = (int8_t*)malloc(mem_size_A);
  h_B = (int8_t*)malloc(mem_size_B);
  h_C = (int8_t*)malloc(mem_size_C);
  h_C_ref = (int8_t*)malloc(mem_size_C);
  h_scale = (float*)malloc(mem_size_scale_bias);
  h_bias = (float*)malloc(mem_size_scale_bias);

  /***********Host array initialization***********/
  for (int i = 0; i < M * K; i += 1) {
    h_A[i] = rand() % 5;
  }
  for (int i = 0; i < K * N; i += 1) {
    h_B[i] = rand() % 5;
  }
  for (int i = 0; i < M * N; i += 1) {
    h_C[i] = 0;
  }
  for (int i = 0; i < N; i += 1) {
    h_scale[i] = rand() % 5;
  }
  for (int i = 0; i < N; i += 1) {
    h_bias[i] = 0;
  }

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

  bool status =
      gemm(&ctx, a_dtype, a_transpose, ptr_A, a_dim0, a_dim1, b_dtype,
           b_transpose, ptr_B, b_dim0, b_dim1, c_dtype, ptr_C, c_dim0, c_dim1,
           batch_count, false, false, &alpha, &beta, scale, bias);
  assert(status == true);

  checkCudaErrors(cudaMemcpy(h_C, ptr_C, mem_size_C, cudaMemcpyDeviceToHost));

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int32_t res = 0;
      for (int k = 0; k < K; ++k) {
        res += h_A[i * K + k] * h_B[j * K + k];
      }
      h_C_ref[i * N + j] =
          int8_t(std::max(float(0), float(res * h_scale[j] + h_bias[j])));
    }
  }

  bool correct = true;
  double eps = 1.e-6;

  for (int i = 0; i < M * N; i++) {
    double abs_err = fabs(h_C[i] - h_C_ref[i]);
    double dot_length = M;
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
}
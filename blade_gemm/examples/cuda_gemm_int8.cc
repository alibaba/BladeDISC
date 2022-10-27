#include <assert.h>
#include <bladnn/backend/cutlass/cutlass_handle.h>
#include <bladnn/bladnn.h>
#include <cuda.h>

using namespace bladnn;

#define checkCudaErrors(func)                                                \
  {                                                                          \
    cudaError_t e = (func);                                                  \
    if (e != cudaSuccess)                                                    \
      printf("%s %d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
  }

void host_gemm(int M, int K, int N, void* A, void* B, void* C, float alpha,
               float beta) {
  /*
  for(int m = 0; m < M; m += 1){
      for(int n = 0; n < N; n += 1){
          int acc = 0;
          for(int k = 0; k < K; k += 1){
              acc += A[]
          }
      }
  }
  */
}

int main() {
  cudaStream_t s;
  cudaStreamCreate(&s);

  Context ctx;
  ctx.stream = s;

  int M = 1024;
  int K = 1024;
  int N = 1024;

  size_t mem_size_A = M * K * sizeof(int8_t);
  size_t mem_size_B = K * N * sizeof(int8_t);
  size_t mem_size_C = M * N * sizeof(int8_t);

  Dtype a_dtype = Dtype::kS8;
  bool a_transpose = 0;
  int8_t *ptr_A, *h_A;
  int a_dim0 = M, a_dim1 = K;

  Dtype b_dtype = Dtype::kS8;
  bool b_transpose = 1;
  int8_t *ptr_B, *h_B;
  int b_dim0 = N, b_dim1 = K;

  Dtype c_dtype = Dtype::kS8;
  int8_t *ptr_C, *h_C;
  int c_dim0 = M, c_dim1 = N;

  int batch_count = 1;

  h_A = (int8_t*)malloc(mem_size_A);
  h_B = (int8_t*)malloc(mem_size_B);
  h_C = (int8_t*)malloc(mem_size_C);

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

  float alpha = 1.0f;
  float beta = 0.0f;
  /***********Host array initialization***********/

  checkCudaErrors(cudaMalloc(&ptr_A, mem_size_A));
  checkCudaErrors(cudaMalloc(&ptr_B, mem_size_B));
  checkCudaErrors(cudaMalloc(&ptr_C, mem_size_C));

  checkCudaErrors(cudaMemcpy(ptr_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(ptr_B, h_B, mem_size_B, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(ptr_C, h_C, mem_size_C, cudaMemcpyHostToDevice));

  bool status = gemm(&ctx, a_dtype, a_transpose, ptr_A, a_dim0, a_dim1, b_dtype,
                     b_transpose, ptr_B, b_dim0, b_dim1, c_dtype, ptr_C, c_dim0,
                     c_dim1, batch_count, false, false, &alpha, &beta);
  assert(status == true);
  printf("pass\n");
}
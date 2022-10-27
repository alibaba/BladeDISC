// nvcc examples/cuda_gemm.cc -Iinclude -lblade_gemm -Lbuild
// -L../../bladedisc/tao/bazel-bin/external/blade_gemm/blade_gemm/lib64/
// -lcublas BLADE_GEMM_TUNE_JIT=1 BLADE_GEMM_VERBOSE=1 ./a.out
#include <cuda_runtime.h>
#include <stdlib.h>

#include "bladnn/bladnn.h"

bool gemm(bladnn::Context& bladnn_ctx, bladnn::Dtype dtype, void* a, void* b,
          void* c, int M, int N, int K, bool tp_a, bool tp_b) {
  bool ret =
      bladnn::gemm(&bladnn_ctx, dtype, tp_a, a, tp_a ? K : M, tp_a ? M : K,
                   dtype, tp_b, b, tp_b ? N : K, tp_b ? K : N, dtype, c, M, N);
  return ret;
}

bool hgemm() {
  cudaStream_t s;
  cudaStreamCreate(&s);
  bladnn::Context bladnn_ctx{s};
  bladnn::Dtype dtype = bladnn::Dtype::kF16;
  void *a, *b, *c;
  int M = 1024;
  int N = 1024;
  int K = 1024;
  cudaMalloc(&c, M * N * 2);
  cudaMalloc(&a, M * K * 2);
  cudaMalloc(&b, K * N * 2);
  bool ret = true;
  ret &= gemm(bladnn_ctx, dtype, a, b, c, M, N, K, false, false);
  return ret;
}

bool hgemm_rand_shapes() {
  int MAX = 16384;
  int MID = 512;
  cudaStream_t s;
  cudaStreamCreate(&s);
  bladnn::Context bladnn_ctx{s};
  bladnn::Dtype dtype = bladnn::Dtype::kF16;
  void *a, *b, *c;
  cudaMalloc(&c, MAX * MAX * 2);
  cudaMalloc(&a, MAX * MAX * 2);
  cudaMalloc(&b, MAX * MAX * 2);
  for (int i = 0; i < 200000; ++i) {
    int M = rand() % MID;
    int N = rand() % MID;
    int K = rand() % MID;
    gemm(bladnn_ctx, dtype, a, b, c, M, N, K, false, false);
  }
  for (int i = 0; i < 200000; ++i) {
    int M = rand() % MID;
    int N = rand() % MID;
    int K = rand() % MAX;
    gemm(bladnn_ctx, dtype, a, b, c, M, N, K, false, false);
  }
  for (int i = 0; i < 200000; ++i) {
    int M = rand() % MID;
    int N = rand() % MAX;
    int K = rand() % MID;
    gemm(bladnn_ctx, dtype, a, b, c, M, N, K, false, false);
  }
  for (int i = 0; i < 200000; ++i) {
    int M = rand() % MAX;
    int N = rand() % MID;
    int K = rand() % MID;
    gemm(bladnn_ctx, dtype, a, b, c, M, N, K, false, false);
  }
  for (int i = 0; i < 200000; ++i) {
    int M = rand() % MAX;
    int N = rand() % MAX;
    int K = rand() % MID;
    gemm(bladnn_ctx, dtype, a, b, c, M, N, K, false, false);
  }
  for (int i = 0; i < 200000; ++i) {
    int M = rand() % MAX;
    int N = rand() % MID;
    int K = rand() % MAX;
    gemm(bladnn_ctx, dtype, a, b, c, M, N, K, false, false);
  }
  for (int i = 0; i < 200000; ++i) {
    int M = rand() % MID;
    int N = rand() % MAX;
    int K = rand() % MAX;
    gemm(bladnn_ctx, dtype, a, b, c, M, N, K, false, false);
  }
  for (int i = 0; i < 200000; ++i) {
    int M = rand() % MAX;
    int N = rand() % MAX;
    int K = rand() % MAX;
    gemm(bladnn_ctx, dtype, a, b, c, M, N, K, false, false);
  }
  return true;
}

int main() {
  hgemm();
  // hgemm_rand_shapes();
  return 0;
}

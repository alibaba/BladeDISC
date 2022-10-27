// hipcc examples/rocm_gemm.cc -Lbuild
// -L../../bladedisc/tao/bazel-bin/external/blade_gemm/blade_gemm/lib64/
// -lblade_gemm -L/opt/rocm-5.1.0/lib/ -lrocblas -Iinclude BLADE_GEMM_TUNE_JIT=1
// BLADE_GEMM_VERBOSE=1 ./a.out
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "bladnn/bladnn.h"

bool gemm(bladnn::Context& bladnn_ctx, bladnn::Dtype dtype, void* a, void* b,
          void* c, int M, int N, int K, bool tp_a, bool tp_b, int batch = 1) {
  bool ret = bladnn::gemm(&bladnn_ctx, dtype, tp_a, a, tp_a ? K : M,
                          tp_a ? M : K, dtype, tp_b, b, tp_b ? N : K,
                          tp_b ? K : N, dtype, c, M, N, batch);
  return ret;
}

bool dgemm() {
  hipStream_t s;
  hipStreamCreate(&s);
  bladnn::Context bladnn_ctx{s};
  bladnn::Dtype dtype = bladnn::Dtype::kF64;
  void *a, *b, *c;
  int batch = 2;
  int M = 8192;
  int N = 1536;
  int K = 240;
  hipMalloc(&c, batch * M * N * sizeof(double));
  hipMalloc(&a, batch * M * K * sizeof(double));
  hipMalloc(&b, batch * K * N * sizeof(double));
  bool ret = true;
  ret &= gemm(bladnn_ctx, dtype, a, b, c, M, N, K, true, false, batch);
  return ret;
}

bool sgemm() {
  hipStream_t s;
  hipStreamCreate(&s);
  bladnn::Context bladnn_ctx{s};
  bladnn::Dtype dtype = bladnn::Dtype::kF32;
  void *a, *b, *c;
  int batch = 100;
  int M = 4;
  int N = 4;
  int K = 12288;
  hipMalloc(&c, batch * M * N * sizeof(float));
  hipMalloc(&a, batch * M * K * sizeof(float));
  hipMalloc(&b, batch * K * N * sizeof(float));
  bool ret = true;
  ret &= gemm(bladnn_ctx, dtype, a, b, c, M, N, K, true, false, batch);
  return ret;
}

int main() {
  dgemm();
  sgemm();
  return 0;
}

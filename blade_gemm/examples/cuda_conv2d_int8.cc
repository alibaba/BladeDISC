#include <assert.h>
#include <bladnn/backend/cutlass/cutlass_handle.h>
#include <bladnn/bladnn.h>
#include <cuda_runtime.h>

using namespace bladnn;

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
int test_case[17][9] = {
    {1, 224, 224, 3, 64, 7, 7, 2, 2},   {1, 56, 56, 64, 64, 1, 1, 1, 1},
    {1, 56, 56, 64, 64, 3, 3, 1, 1},    {1, 56, 56, 64, 256, 1, 1, 1, 1},
    {1, 56, 56, 256, 64, 1, 1, 1, 1},   {1, 56, 56, 256, 128, 1, 1, 2, 2},
    {1, 28, 28, 128, 128, 3, 3, 1, 1},  {1, 28, 28, 128, 512, 1, 1, 1, 1},
    {1, 28, 28, 512, 128, 1, 1, 1, 1},  {1, 28, 28, 512, 256, 1, 1, 2, 2},
    {1, 14, 14, 256, 256, 3, 3, 1, 1},  {1, 14, 14, 256, 1024, 1, 1, 1, 1},
    {1, 14, 14, 1024, 256, 1, 1, 1, 1}, {1, 14, 14, 1024, 512, 1, 1, 2, 2},
    {1, 7, 7, 512, 512, 3, 3, 1, 1},    {1, 7, 7, 512, 2048, 1, 1, 1, 1},
    {1, 7, 7, 2048, 512, 1, 1, 1, 1}};

void run_conv2d(int N, int H, int W, int C, int K, int R, int S, int stride_h,
                int stride_w, int pad_h = 1, int pad_w = 1, int dilation_h = 1,
                int dilation_w = 1, int groups = 1) {
  cudaStream_t s;
  cudaStreamCreate(&s);

  ConvKind kind = ConvKind::kFprop;
  Layout data_layout = Layout::kNHWC;
  Layout kernel_layout = Layout::kNHWC;

  Dtype in_dtype = Dtype::kS8;
  Dtype out_dtype = Dtype::kS8;

  int P = getFwdConvOutputDim(H, pad_w, R, stride_h, dilation_h);
  int Q = P;

  printf("P: %d, Q: %d\n", P, Q);

  size_t mem_size_A = N * H * W * C * sizeof(int8_t);
  size_t mem_size_B = K * R * S * C * sizeof(int8_t);
  size_t mem_size_C = N * P * Q * K * sizeof(int8_t);

  int8_t *ptr_A, *h_A, *ptr_B, *h_B, *ptr_C, *h_C;

  h_A = (int8_t*)malloc(mem_size_A);
  h_B = (int8_t*)malloc(mem_size_B);
  h_C = (int8_t*)malloc(mem_size_C);

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

  float alpha = 1.0f;
  float beta = 0.0f;
  /***********Host array initialization***********/

  checkCudaErrors(cudaMalloc(&ptr_A, mem_size_A));
  checkCudaErrors(cudaMalloc(&ptr_B, mem_size_B));
  checkCudaErrors(cudaMalloc(&ptr_C, mem_size_C));

  checkCudaErrors(cudaMemcpy(ptr_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(ptr_B, h_B, mem_size_B, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(ptr_C, h_C, mem_size_C, cudaMemcpyHostToDevice));

  bool status =
      conv2d(s, in_dtype, out_dtype, kind, data_layout, kernel_layout, N, H, W,
             C, K, R, S, P, Q, pad_h, pad_w, stride_h, stride_w, dilation_h,
             dilation_w, groups, &alpha, ptr_A, ptr_B, &beta, ptr_C, ptr_C);

  assert(status == true);
  printf("pass\n");
}

int main() {
  for (int i = 0; i < 17; i += 1) {
    run_conv2d(test_case[i][0], test_case[i][1], test_case[i][2],
               test_case[i][3], test_case[i][4], test_case[i][5],
               test_case[i][6], test_case[i][7], test_case[i][8]);
    printf("conv %d tested\n", i);
  }
  return 0;
}
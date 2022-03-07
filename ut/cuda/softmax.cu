#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <stdio.h>

__global__ void softmax1st(float* input,  float* mul,
                               float* param,  float* output) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  // blockDim.x == 256
  int output_dim = blockIdx.x;
  int thread_id = threadIdx.x;
  int warp_id = thread_id / 32;
  int lane_id = thread_id % 32;

  float local_max = -INFINITY;
  for (int64_t i = thread_id; i < 1024; i += blockDim.x) {
      float tmp = input[output_dim * 1024 + i] * param[i];
      mul[output_dim * 1024 + i] = tmp;
      if (tmp > local_max) {
          local_max = tmp;
      }
  }

  __shared__ float s_data[32];

  for (int64_t i = 1; i < 32; i *= 2) {
    float tmp = __shfl_xor_sync(unsigned(-1), local_max, i);
    if (tmp < local_max) {
        local_max = tmp;
    }
  }
  if (lane_id == 0) {
    s_data[warp_id] = local_max;
    s_data[warp_id] = local_max;
  }
  __syncthreads();
  float max = -INFINITY;
  if (lane_id < 8) {
    max = s_data[lane_id];
  }

  for (int64_t i = 1; i < 8; i *= 2) {
    float tmp =
        __shfl_xor_sync(unsigned(-1), max, i);
    if (tmp > max) {
        max = tmp;
    }
  }
  if (thread_id == 0) {
      output[output_dim] = max;
  }
}


int main() {
  const int d0 = 1;
  const int d1 = 256;
  const int d2 = 768;
  float *input, *mul, *param, *output;
  uint64_t mem_size = 1024 * sizeof(float);
  checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&input), mem_size));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&output), mem_size));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&mul), mem_size));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&param), mem_size));
  float* h_init = reinterpret_cast<float*>(malloc(mem_size));
  for (int64_t i = 0; i < 1024; i++) {
    h_init[i] = 0.25;
  }
  checkCudaErrors(cudaMemcpy(input, h_init, mem_size, cudaMemcpyDefault));
  checkCudaErrors(cudaMemcpy(param, h_init, mem_size, cudaMemcpyDefault));
  checkCudaErrors(cudaMemset(output, 0, mem_size));
  checkCudaErrors(cudaMemset(mul, 0, mem_size));

  free(h_init);
//   float yita = 0.0001;
//   float gamma = 1.3;
//   float beta = 1.3;

  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  // warmup
  checkCudaErrors(cudaEventRecord(start));
  int times = 20;
  for (int i = 0; i < times; i++)
    softmax1st<<<1, 256>>>(input, mul, param, output);
  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));
  float msec = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msec, start, stop));
  printf("execution time = %f\n", msec);
  return 0;
}

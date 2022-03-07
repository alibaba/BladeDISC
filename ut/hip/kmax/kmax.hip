#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
//#include <helper_hip.h>
//#include <helper_functions.h>
#include <stdio.h>

#define checkCudaErrors(val) \
	if (val != hipSuccess) exit(EXIT_FAILURE)

#define D1 4096
#define D0 1

__global__ void softmax1st(float* input,  
                                float* output, int d1) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  // blockDim.x == 256
  int output_dim = blockIdx.x;
  int thread_id = threadIdx.x;
  int warp_id = thread_id / 32;
  int lane_id = thread_id % 32;

  float local_max = -INFINITY;
  for (int64_t i = thread_id; i < D1; i += blockDim.x) {
      float tmp = input[output_dim * D1 + i];
      if (tmp > local_max) {
          local_max = tmp;
      }
  }

  __shared__ float s_data[32];

  for (int64_t i = 1; i < 32; i *= 2) {
    float tmp = __shfl_xor(local_max, i, 32);
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
        __shfl_xor(max, i, 32);
    if (tmp > max) {
        max = tmp;
    }
  }
  if (thread_id == 0) {
      output[output_dim] = max;
  }
}


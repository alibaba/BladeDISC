#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
//#include <helper_hip.h>
//#include <helper_functions.h>
#include <stdio.h>

#define checkCudaErrors(val) \
	if (val != hipSuccess) exit(EXIT_FAILURE)

#define D1 4096
#define D0 1
#define warp_size  32
#define launch_dim (warp_size*8)

 __global__ void main_kRowReduction_reduce__2_1_0___1b1rX_vectile2X_no_vectile(float* arg0, float* input, int arg2, int  arg3, int arg4, int arg5, int arg6,
                               int arg7, int arg8,float* arg9, float* output, int arg11, int arg12, int arg13) {
//  int arg7 = blockDim.x;
  int index = blockIdx.x * arg7 + threadIdx.x;
  // blockDim.x == 256
  if (index < arg8) { 
    int idx = blockIdx.x * arg7 + threadIdx.x;
    int thread_id = idx % launch_dim;
    int block_id = idx / launch_dim;
    int warp_id = thread_id / warp_size;
    int lane_id = thread_id % warp_size;
    bool lane_0 = lane_id == 0;
    bool warp_0 = warp_id == 0;

    float local_max = -INFINITY;
    for (int64_t i = thread_id; i < arg4; i += launch_dim) {
        float tmp = input[block_id * arg4 + i];
        float local_tmp;
        if (local_max > tmp) {
          local_tmp = local_max; 
        } else {
          local_tmp = tmp;
        }
        if (isnan(tmp)) {
          local_max = tmp; 
        } else {
          local_max = local_tmp;           
        }
    }

    __shared__ float s_data[warp_size];

    #pragma unroll
    for (int64_t i = 1; i < warp_size; i *= 2) {
      float tmp = __shfl_xor(local_max, i, warp_size);
      float local_tmp;
      if (local_max > tmp) {
          local_tmp = local_max; 
        } else {
          local_tmp = tmp;
        }
        if (isnan(tmp)) {
          local_max = tmp; 
        } else {
          local_max = local_tmp;           
      }
    }
    if (lane_0) {
      s_data[warp_id] = local_max;
    }
    __syncthreads();
    float max;
    if (lane_id < 8) {
      max = s_data[lane_id];
    } else {
      max = -INFINITY;
    }
    #pragma unroll
    for (int64_t i = 1; i < 8; i *= 2) {
      float tmp =
          __shfl_xor(max, i, warp_size);
      float local_tmp;
      if (local_max > tmp) {
          local_tmp = local_max; 
        } else {
          local_tmp = tmp;
        }
        if (isnan(tmp)) {
          max = tmp; 
        } else {
          max = local_tmp;           
      }
    }
//    mid[thread_id] = max;
    if (thread_id == 0) {
        output[block_id] = max;
    }
  }
}


__global__ void softmax1st_simple(float* input,  
                                float* output, int d1) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  // blockDim.x == 256
  int output_dim = blockIdx.x;
  int thread_id = threadIdx.x;
  int warp_id = thread_id / warp_size;
  int lane_id = thread_id % warp_size;

  float local_max = -INFINITY;
  for (int64_t i = thread_id; i < d1; i += blockDim.x) {
      float tmp = input[output_dim * d1 + i];
      if (tmp > local_max) {
          local_max = tmp;
      }
  }

  __shared__ float s_data[warp_size];

  for (int64_t i = 1; i < warp_size; i *= 2) {
    float tmp = __shfl_xor(local_max, i, warp_size);
    if (tmp > local_max) {
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
        __shfl_xor(max, i, warp_size);
    if (tmp > max) {
        max = tmp;
    }
  }
  if (thread_id == 0) {
      output[output_dim] = max;
  }
}



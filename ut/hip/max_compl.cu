#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
//#include <helper_hip.h>
//#include <helper_functions.h>
#include <stdio.h>

#define checkCudaErrors(val) \
	if (val != hipSuccess) exit(EXIT_FAILURE)

#define D1 1024
#define D0 1
#define warp_size  32
#define launch_dim warp_size*8
 __global__ void softmax1st(float* input,
                                float* output, int arg4, int arg8) {
  int arg7 = blockDim.x;
  int index = blockIdx.x * arg7 + threadIdx.x;
  // blockDim.x == 256
  if (index < arg8) { 
    int idx = blockIdx.x * arg7 + threadIdx.x;
    int thread_id = idx % 256;
    int block_id = idx / 256;
    int warp_id = thread_id / warp_size;
    int lane_id = thread_id % warp_size;
    bool lane_0 = lane_id == 0;
    bool warp_0 = warp_id == 0;

    float local_max = -INFINITY;
    for (int64_t i = thread_id; i < arg4; i += blockDim.x) {
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

    //#pragma unroll
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
    //#pragma unroll
    for (int64_t i = 1; i < 8; i *= 2) {
      float tmp =
          __shfl_xor(max, i, warp_size);
      float local_tmp;
      if (max > tmp) {
          local_tmp = max; 
        } else {
          local_tmp = tmp;
        }
        if (isnan(tmp)) {
          max = tmp; 
        } else {
          max = local_tmp;           
      }
    }
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
  for (int64_t i = thread_id; i < d1; i += launch_dim) {
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


int main() {
  const int d0 = D0;
  const int d1 = D1;
  float *input, *output, *mid;
  uint64_t mem_size = d0 * d1 * sizeof(float);
  uint64_t mem_size0 = d0 * sizeof(float);
  checkCudaErrors(hipMalloc(reinterpret_cast<void**>(&input), mem_size));
  checkCudaErrors(hipMalloc(reinterpret_cast<void**>(&output), mem_size0));
  checkCudaErrors(hipMalloc(reinterpret_cast<void**>(&mid), mem_size));
  float* h_init = reinterpret_cast<float*>(malloc(mem_size));
  for (int64_t i = 0; i < mem_size/sizeof(float) ; i++) {
    h_init[i] = 0.001*i;
//    printf("%f ", h_init[i]);
  }
  printf("\n");
  checkCudaErrors(hipMemcpy(input, h_init, mem_size, hipMemcpyDefault));
  checkCudaErrors(hipMemset(output, 0, mem_size0));
  checkCudaErrors(hipMemset(mid, 0, mem_size));

  free(h_init);
//   float yita = 0.0001;
//   float gamma = 1.3;
//   float beta = 1.3;

  hipEvent_t start, stop;
  checkCudaErrors(hipEventCreate(&start));
  checkCudaErrors(hipEventCreate(&stop));
  float* o_init = reinterpret_cast<float*>(malloc(mem_size));
  memset(o_init, 0, mem_size0);

  // warmup
  checkCudaErrors(hipEventRecord(start));
  int times = 100;
  for (int i = 0; i < times; i++) {
	  /*
  	for (int64_t j = 0; j < mem_size/sizeof(float) ; j++) {
    		h_init[j] = 0.001*i;
  	}
        checkCudaErrors(hipMemcpy(input, h_init, mem_size, hipMemcpyDefault));*/
	softmax1st<<<d0, launch_dim>>>(input, output, d1, d0*launch_dim);
        softmax1st_simple<<<d0, launch_dim>>>(input, output, d1);
	/*
  	checkCudaErrors(hipMemcpy(o_init, output, mem_size0, hipMemcpyDefault));*/
  }
  checkCudaErrors(hipEventRecord(stop));
  checkCudaErrors(hipEventSynchronize(stop));
  float msec = 0.0f;
  checkCudaErrors(hipEventElapsedTime(&msec, start, stop));
  printf("execution time = %f\n", msec);
  float* o_mid = reinterpret_cast<float*>(malloc(mem_size));
  memset(o_mid, 0, mem_size);
  checkCudaErrors(hipMemcpy(o_mid, mid, mem_size, hipMemcpyDefault));
  for (int64_t i = 0; i < d0; i++) {
	 printf(" %f ", o_init[i]);
  }
  printf("\n");/*
  for (int64_t i = 0; i < d0*d1; i++) {
	 printf(" %f ", o_mid[i]);
  }
  printf("\n");*/



  return 0;
}

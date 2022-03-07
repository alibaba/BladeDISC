#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <stdio.h>

#define checkCudaErrors(val) \
	if (val != hipSuccess) exit(EXIT_FAILURE)

#define D1 128
#define D0 4096
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


__global__ void copy(float* input, float* output) {
   output[blockIdx.x * blockDim.x + threadIdx.x]  = input[blockIdx.x * blockDim.x + threadIdx.x];
}

int main() {
  const int d0 = D0;
  const int d1 = D1;
  float *input, *output;
  float *inputt, *outputt;
  uint64_t mem_size = d0 * d1 * sizeof(float);
  uint64_t mem_size0 = d0 * sizeof(float);
  checkCudaErrors(hipMalloc(reinterpret_cast<void**>(&input), mem_size));
  checkCudaErrors(hipMalloc(reinterpret_cast<void**>(&inputt), mem_size));
  checkCudaErrors(hipMalloc(reinterpret_cast<void**>(&outputt), mem_size0));
  checkCudaErrors(hipMalloc(reinterpret_cast<void**>(&output), mem_size0));
  float* h_init = reinterpret_cast<float*>(malloc(mem_size));
  checkCudaErrors(hipMemset(output, 0, mem_size0));
  checkCudaErrors(hipMemset(inputt, 0, mem_size));

//   float yita = 0.0001;
//   float gamma = 1.3;
//   float beta = 1.3;

  hipEvent_t start, stop;
  checkCudaErrors(hipEventCreate(&start));
  checkCudaErrors(hipEventCreate(&stop));
  float* o_init = reinterpret_cast<float*>(malloc(mem_size0));
  memset(o_init, 0, mem_size0);

  // warmup
  checkCudaErrors(hipEventRecord(start));
  int times = 100;
  for (int64_t j = 0; j < mem_size/sizeof(float) ; j++) {
    h_init[j] = 0.001 * (j % D1);
  }
  printf("aaaaa\n");
  for (int i = 0; i < times; i++) {
  checkCudaErrors(hipMemcpy(input, h_init, mem_size, hipMemcpyDefault));
    copy<<<d0*d1/launch_dim, launch_dim>>>(input, inputt);
    main_kRowReduction_reduce__2_1_0___1b1rX_vectile2X_no_vectile<<<d0/PARA, launch_dim>>>(inputt, inputt, 0, D0, D1, D1,
		                        1, launch_dim, D0/PARA*launch_dim, outputt, outputt,0, D0, 1);
    copy<<<1, d0>>>(outputt, output);
    checkCudaErrors(hipMemcpy(o_init, output, mem_size0, hipMemcpyDefault));
  }
    for (int64_t j = 0; j < d0; j++) {
	 printf("output: %f\n", o_init[j]);
    }
  checkCudaErrors(hipEventRecord(stop));
  checkCudaErrors(hipEventSynchronize(stop));
  float msec = 0.0f;
  checkCudaErrors(hipEventElapsedTime(&msec, start, stop));
  printf("execution time = %f\n", msec);


  free(h_init);

  return 0;
}

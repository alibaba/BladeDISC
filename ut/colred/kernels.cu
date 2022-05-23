
// arg0 = 25
// arg1 = [11776, 25]
// arg2 = 256
// arg3 = 376832
// arg4 = 4
// arg5 = [11776, 25]
// arg6 = 11776*25
// arg7 = 0
// arg8 = 0
// arg9 = 1
// arg10 = 1
// arg11 = 50
// arg12 = 11776
// arg13 = [11776, 50]
// arg14 = 0
// arg15 = 25
// arg16 = [11776, 25]
// arg17 = [11776, 25]
// arg18 = [11776, 25]
// arg19 = 11776
// arg20 = 25
// arg21 = [11776, 25]
// arg22 = [25]
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <stdio.h>

#define checkCudaErrors(val) \
	if (val != hipSuccess) exit(EXIT_FAILURE)


#define D0 11776
#define D1 25
#define D2 50

template<typename T>
__global__ void kernel(T* arg1, T* arg5, T* arg16, T* arg17, T* arg18, T* arg13, T* arg21, T* arg22) {
     __shared__ T arg23[256];
    int v14 = blockIdx.x * 256; //arg2 256 threaddim
    int v15 = threadIdx.x;
    int total_threads = v14 + v15;
    if (total_threads < 376832) {
        int thread_x = threadIdx.x; //v16 % 256;
        int v23 = threadIdx.x / 8; // v32
        int v24 = threadIdx.x % 8; // v32
        int v25 = blockIdx.x / 4;
        int v26 = blockIdx.x % 4;
        int v28 = v25 * 32 + v23; // limit 368
        int v30 = v26 * 8 + v24; // limit 32 -> 25
        bool v31 = v28 < 11776;
        bool v32 = v30 < 25;
        if (v31 && v32) {
            int v53 = v28 * 25 + v30;
            T v55 = arg5[v53]; // total arg6
            T v64 = arg13[v28 * 50 + v30]; // total arg12 * arg11
            T v70 = arg13[v28 * 50 + v30 + 25];
            T v71 = v55 + v64 + v70;
            T v73 = arg1[v53]; //total arg6
            T v76 = arg16[v53];//total arg6
            T v78 = arg17[v53];//total arg6
            T v81 = -2 * v76 * v78 * v73 + v71;
            T v83 = arg18[v53]; // total arg6
            T v84 = v83 * v81;
            arg21[v53] = v84; // total arg19 * 
            arg23[thread_x] = v84;
        } else {
            arg23[thread_x] = 0;
        }
        __syncthreads();
        if (threadIdx.x < 128 && v28 + 16 < 11776) {
            arg23[thread_x] = arg23[thread_x] + arg23[thread_x + 128];
        }
        __syncthreads();
        if (threadIdx.x < 64 && v28 + 8 < 11776) {
            arg23[thread_x] = arg23[thread_x] + arg23[thread_x + 64];
        }
        __syncthreads();
        if (threadIdx.x < 32 && v28 + 4 < 11776) {
            arg23[thread_x] = arg23[thread_x] + arg23[thread_x + 32];
        }
        __syncthreads();
        if (threadIdx.x < 16 && v28 + 2 < 11776) {
            arg23[thread_x] = arg23[thread_x] + arg23[thread_x + 16];
        }
        __syncthreads();
        if (v23 == 0 && v31 && v32) {
            atomicAdd(&arg22[v30], arg23[thread_x] + arg23[thread_x + 8]);
        }
    }
}

// template __global__ void kernel<double>();


int main() {
  const int d0 = D0;
  const int d1 = D1;
  const int d2 = D2;
  double* ptr[6]; 
  double* output, *output0;
  uint64_t mem_size = d0 * d1 * sizeof(double);
  uint64_t mem_size0 = d0 * d2 * sizeof(double);
  uint64_t mem_size1 = d2 * sizeof(double);
  

  for (int i = 0; i < 5; i++) {
    checkCudaErrors(hipMalloc(reinterpret_cast<void**>(&ptr[i]), mem_size));
  }
  checkCudaErrors(hipMalloc(reinterpret_cast<void**>(&ptr[5]), mem_size0));

  checkCudaErrors(hipMalloc(reinterpret_cast<void**>(&output), mem_size));
  checkCudaErrors(hipMalloc(reinterpret_cast<void**>(&output0), mem_size1));
//   float* h_init = reinterpret_cast<float*>(malloc(mem_size));
//   for (int64_t i = 0; i < mem_size/sizeof(float) ; i++) {
//     h_init[i] = 0.001*i;
// //    printf("%f ", h_init[i]);
//   }
//   printf("\n");
//   checkCudaErrors(hipMemcpy(input, h_init, mem_size, hipMemcpyDefault));
//   checkCudaErrors(hipMemset(output, 0, mem_size0));
//   checkCudaErrors(hipMemset(mid, 0, mem_size));

//   free(h_init);
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
  int times = 1;
  for (int i = 0; i < times; i++) {
	  /*
  	for (int64_t j = 0; j < mem_size/sizeof(float) ; j++) {
    		h_init[j] = 0.001*i;
  	}
        checkCudaErrors(hipMemcpy(input, h_init, mem_size, hipMemcpyDefault));*/
	kernel<double><<<1472, 256>>>(ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], output, output0);
        // softmax1st_simple<<<d0, launch_dim>>>(input, output, d1);
	/*
  	checkCudaErrors(hipMemcpy(o_init, output, mem_size0, hipMemcpyDefault));*/
  }
  checkCudaErrors(hipEventRecord(stop));
  checkCudaErrors(hipEventSynchronize(stop));
  float msec = 0.0f;
  checkCudaErrors(hipEventElapsedTime(&msec, start, stop));
  printf("execution time = %f\n", msec / times);
//   float* o_mid = reinterpret_cast<float*>(malloc(mem_size));
//   memset(o_mid, 0, mem_size);
//   checkCudaErrors(hipMemcpy(o_mid, mid, mem_size, hipMemcpyDefault));
//   for (int64_t i = 0; i < d0; i++) {
// 	 printf(" %f ", o_init[i]);
//   }
// //   printf("\n");/*
//   for (int64_t i = 0; i < d0*d1; i++) {
// 	 printf(" %f ", o_mid[i]);
//   }
//   printf("\n");*/



  return 0;
}


















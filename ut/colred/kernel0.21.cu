
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
#define T double

// template<typename T>
__global__ void kernel(
    uint64_t arg0,
    T* arg1_0, T* arg1, uint64_t arg1_1, uint64_t arg1_2, uint64_t arg1_3, uint64_t arg1_4, uint64_t arg1_5,
    uint64_t arg2, uint64_t arg3, uint64_t arg4,
    T* arg50, T* arg5, uint64_t arg51, uint64_t arg52, uint64_t arg53, uint64_t arg54, uint64_t arg55, 
    uint64_t arg6, uint64_t arg7, uint64_t arg8, uint64_t arg9, uint64_t arg10, uint64_t arg11, uint64_t arg12,
    T* arg130, T* arg13, uint64_t arg131, uint64_t arg132, uint64_t arg133, uint64_t arg134, uint64_t arg135,
    uint64_t arg14, uint64_t arg15,
    T* arg160, T* arg16, uint64_t arg161, uint64_t arg162, uint64_t arg163, uint64_t arg164, uint64_t arg165,
    T* arg170, T* arg17, uint64_t arg171, uint64_t arg172, uint64_t arg173, uint64_t arg174, uint64_t arg175,
    T* arg180, T* arg18, uint64_t arg181, uint64_t arg182, uint64_t arg183, uint64_t arg184, uint64_t arg185,
    uint64_t arg19, uint64_t arg20,
    T* arg210, T* arg21, uint64_t arg211, uint64_t arg212, uint64_t arg213, uint64_t arg214, uint64_t arg215,
    T* arg220, T* arg22, uint64_t arg221, uint64_t arg222, uint64_t arg223) {
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



// template<> __global__ void kernel<double>(
//     uint64_t arg0,
//     double* arg1_0, double* arg1, uint64_t arg1_1, uint64_t arg1_2, uint64_t arg1_3, uint64_t arg1_4, uint64_t arg1_5,
//     uint64_t arg2, uint64_t arg3, uint64_t arg4,
//     double* arg50, double* arg5, uint64_t arg51, uint64_t arg52, uint64_t arg53, uint64_t arg54, uint64_t arg55, 
//     uint64_t arg6, uint64_t arg7, uint64_t arg8, uint64_t arg9, uint64_t arg10, uint64_t arg11, uint64_t arg12,
//     double* arg130, double* arg13, uint64_t arg131, uint64_t arg132, uint64_t arg133, uint64_t arg134, uint64_t arg135,
//     uint64_t arg14, uint64_t arg15,
//     double* arg160, double* arg16, uint64_t arg161, uint64_t arg162, uint64_t arg163, uint64_t arg164, uint64_t arg165,
//     double* arg170, double* arg17, uint64_t arg171, uint64_t arg172, uint64_t arg173, uint64_t arg174, uint64_t arg175,
//     double* arg180, double* arg18, uint64_t arg181, uint64_t arg182, uint64_t arg183, uint64_t arg184, uint64_t arg185,
//     uint64_t arg19, uint64_t arg20,
//     double* arg210, double* arg21, uint64_t arg211, uint64_t arg212, uint64_t arg213, uint64_t arg214, uint64_t arg215,
//     double* arg220, double* arg22, uint64_t arg221, uint64_t arg222, uint64_t arg223
// );















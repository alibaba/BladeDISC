
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
    T* arg0, T* arg1, uint64_t arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5, uint64_t arg6,
    uint64_t arg7, uint64_t arg8, uint64_t arg9, uint64_t arg10, 
    T* arg11, T* arg12, uint64_t arg13, uint64_t arg14, uint64_t arg15) {
     __shared__ T arg23[256];
    int total_threads = blockIdx.x * arg7 + threadIdx.x;
    if (total_threads < arg8) {
        int thread_x = total_threads % 256; //v16 % 256;
        int block_x = total_threads / 256;
        int v41 = thread_x / 8; // v32
        int v42 = thread_x % 8; // v32
        int v43 = block_x / arg9;
        int v44 = block_x % arg9;
        int v46= v43 * 32 + v41; // limit 368
        int v48 = v44 * 8 + v42; // limit 32 -> 25
        bool v31 = v46 < arg3;
        bool v32 = v48 < arg4;
        if (v31 && v32) {
            int v54 = v46 * arg4 + v48;
            T v55 = arg1[v54]; // total arg6
            arg23[thread_x] = v55;
        } else {
            arg23[thread_x] = 0;
        }
        __syncthreads();
        if (v41 < 16 && v46 + 16 < arg3) {
            arg23[thread_x] = arg23[thread_x] + arg23[thread_x + 128];
        }
        __syncthreads();
        if (v41 < 8 && v46 + 8 < arg3) {
            arg23[thread_x] = arg23[thread_x] + arg23[thread_x + 64];
        }
        __syncthreads();
        if (v41 < 4 && v46 + 4 < arg3) {
            arg23[thread_x] = arg23[thread_x] + arg23[thread_x + 32];
        }
        __syncthreads();
        if (v41 < 2 && v46 + 2 < arg3) {
            arg23[thread_x] = arg23[thread_x] + arg23[thread_x + 16];
        }
        __syncthreads();
        if (v41 == 0 && v31 && v32) {
            atomicAdd(&arg12[v48], arg23[thread_x] + arg23[thread_x + 8]);
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















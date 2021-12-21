#ifndef AI_COMPILER_CUSTOM_LIB_DYN_SORT_H_
#define AI_COMPILER_CUSTOM_LIB_DYN_SORT_H_

#if GOOGLE_CUDA
#include <cuda_runtime.h>
using gpuStream_t = cudaStream_t;
#elif TENSORFLOW_USE_ROCM
#include "rocm/include/hip/hip_runtime.h"
using gpuStream_t = hipStream_t;
#endif

template <typename T>
struct LaunchTfTopKFunctor {
  void operator()(gpuStream_t stream, const T* input, int batch_size,
                  int length, int k, bool sorted, T* output, int* indices);
};

#endif  // AI_COMPILER_CUSTOM_LIB_DYN_SORT_H_

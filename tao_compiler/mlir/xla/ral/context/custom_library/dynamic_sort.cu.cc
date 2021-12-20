#include "tensorflow/compiler/mlir/xla/ral/context/custom_library/dynamic_sort.h"
#include "tensorflow/compiler/mlir/xla/ral/context/custom_library/tf_topk.cu.h"

// tensorflow top-k impl
template <typename T>
void LaunchTfTopKFunctor<T>::operator()(gpuStream_t stream, const T* input,
                                        int batch_size, int length, int k,
                                        bool sorted, T* output, int* indices) {
  impl::LaunchTopKKernel(stream, 0 /* default num_shards */, input, batch_size,
                         length, k, sorted, output, indices);
}

#define TF_TOP_K_BUILDER(T) template struct LaunchTfTopKFunctor<T>;
TF_TOP_K_BUILDER(float);
TF_TOP_K_BUILDER(int);
#undef TF_TOP_K_BUILDER

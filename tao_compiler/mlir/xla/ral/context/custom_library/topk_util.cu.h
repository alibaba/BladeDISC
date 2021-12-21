#ifndef DYN_TOP_K_INTERNAL_H_
#define DYN_TOP_K_INTERNAL_H_

#include <cassert>
#include <limits>

static const unsigned full_mask = 0xffffffff;

#include "atomic.cu.h"
#include "bitonic_sort.cu.h"
#include "warp.cu.h"

template <typename Itype>
__device__ __inline__ static Itype iceil(const Itype dividend,
                                         const Itype divisor) {
  return (dividend + divisor - 1) / divisor;
}

template <typename Dtype, unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA) __global__
    void batchedSetToLowest(const unsigned batchSize, const unsigned width,
                            const unsigned stride, Dtype* array) {
  for (unsigned b = blockIdx.x; b < batchSize; b += gridDim.x) {
    for (unsigned x = threadIdx.x; x < width; x += nthdsPerCTA) {
      array[b * stride + x] = std::numeric_limits<Dtype>::lowest();
    }
  }
}

#endif  // DYN_TOP_K_INTERNAL_H_

// Copyright 2021 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

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

#ifndef DYN_TOP_K_KTH_BIGGEST_H_
#define DYN_TOP_K_KTH_BIGGEST_H_

#include "bitonic_sort.cu.h"
#include "topk_util.cu.h"

// if inData==outData, then it's in-place sorting
// #CTA == batchSize
template <typename Dtype, unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA) __global__
    void batchedKthBiggestKernel(const unsigned batchSize,
                                 const unsigned iWidth, const unsigned k,
                                 Dtype* inData, Dtype* outData) {
  Dtype threadBuf[8];

  threadBuf[0] = threadBuf[1] = threadBuf[2] = threadBuf[3] = threadBuf[4] =
      threadBuf[5] = threadBuf[6] = threadBuf[7] =
          std::numeric_limits<Dtype>::lowest();

  const unsigned batch = (blockIdx.x * nthdsPerCTA + threadIdx.x) >> 5;
  const Dtype* const inDataOff = inData + batch * iWidth;
  const unsigned sortRange = k * 2;

  if (batch < batchSize) {
// assumed k<=128
#pragma unroll
    for (unsigned i = threadIdx.x & 31, j = 0; i < sortRange && j < 8;
         i += 32, ++j) {
      threadBuf[j] = inDataOff[i];
    }

    // sorted:
    // threadIdx.x==0:  255, 223, 191, 159, 127, 95, 63, 31
    // threadIdx.x==1:  254, 222, 190, 158, 126, 94, 62, 30
    // ...
    // threadIdx.x==31: 224, 192, 160, 128, 96,  64, 32, 0
    bitonicSort256_256(threadBuf, full_mask);

    if ((threadIdx.x & 31) == ((k - 1) & 31)) {
      outData[batch] = threadBuf[(k - 1) >> 5];
    }

  }  // if:batch
}

template <typename Dtype>
void batchedKthBiggestGpu(cudaStream_t stream, const unsigned batchSize,
                          const unsigned iWidth, const unsigned k,
                          Dtype* inData, Dtype* outData) {
  // BS: 128 = 4 warps
  const unsigned BS = 128;
  const unsigned GS = (batchSize + 3) / 4;
  batchedKthBiggestKernel<Dtype, BS>
      <<<GS, BS, 0, stream>>>(batchSize, iWidth, k, inData, outData);
}

#endif  // DYN_TOP_K_KTH_BIGGEST_H_

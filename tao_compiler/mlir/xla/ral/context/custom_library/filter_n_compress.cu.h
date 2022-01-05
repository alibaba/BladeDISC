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

#ifndef DYN_TOP_K_FILTER_K_COMPRESS_H_
#define DYN_TOP_K_FILTER_K_COMPRESS_H_

#include "topk_util.cu.h"

// inThresh is a device pointer of length batchSize
// initial values of outNumLeft should be zero
// outIdx are the local batch-wise zero-based indices
// a warp is dedicated to a batch
template <typename KeyType, unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA) __global__
    void batchedFilterNCompressKernel(const unsigned batchSize,
                                      const unsigned iWidth,
                                      const unsigned oWidth, const unsigned k,
                                      KeyType* inKey, KeyType* inThresh,
                                      KeyType* outKey, unsigned* outIdx,
                                      unsigned* outNumLeft) {
  __shared__ KeyType sKey[nthdsPerCTA];
  __shared__ unsigned sIdx[nthdsPerCTA];

  const unsigned gid = blockIdx.x * nthdsPerCTA + threadIdx.x;
  const unsigned nWarps = (nthdsPerCTA * gridDim.x) >> 5;
  const unsigned wid = gid >> 5;

  const unsigned remainder = nWarps % batchSize;
  const unsigned roundUp = iceil(nWarps, batchSize);
  const unsigned roundDw = nWarps / batchSize;

  const unsigned warpBatchId =
      (wid / roundUp) < remainder ? wid / roundUp : (wid - remainder) / roundDw;
  const unsigned batchInitWarp = warpBatchId < remainder
                                     ? roundUp * warpBatchId
                                     : roundDw * warpBatchId + remainder;
  const unsigned batchNumThreads =
      warpBatchId < remainder ? (roundUp << 5) : (roundDw << 5);

  const unsigned lane = threadIdx.x & 31;
  KeyType* const sKeyOff = &sKey[threadIdx.x >> 5 << 5];
  unsigned* const sIdxOff = &sIdx[threadIdx.x >> 5 << 5];
  KeyType key;
  unsigned idx, len, outOff;

  int cond1 = ((gid - (batchInitWarp << 5)) < iWidth);
  const unsigned mask1 = __ballot_sync(full_mask, cond1);
  for (unsigned i = gid - (batchInitWarp << 5); i < iWidth;
       i += batchNumThreads) {
    key = (inKey + warpBatchId * iWidth)[i];
    idx = i;
    const unsigned mask2 = __ballot_sync(mask1, (i < iWidth));
    len = warpFilterCompress(key, idx, inThresh[warpBatchId], sKeyOff, sIdxOff,
                             mask2);
    int valid = (lane < len) ? 1 : 0;
    const unsigned valid_mask = __ballot_sync(full_mask, valid);
    if (valid) {
      if (lane == 0) outOff = atomicAdd(&outNumLeft[warpBatchId], len);
      outOff = __shfl_sync(valid_mask, outOff, 0);
      // TODO: break when necessary
      (outKey + warpBatchId * oWidth)[(outOff + lane) % oWidth] = sKeyOff[lane];
      (outIdx + warpBatchId * oWidth)[(outOff + lane) % oWidth] = sIdxOff[lane];
    }
  }
}

template <typename KeyType>
void batchedFilterNCompressGpu(cudaStream_t stream, const unsigned batchSize,
                               const unsigned iWidth, const unsigned oWidth,
                               const unsigned k, KeyType* inKey,
                               KeyType* inThresh, KeyType* outKey,
                               unsigned* outIdx, unsigned* outNumLeft) {
  cudaMemsetAsync(outNumLeft, 0, batchSize * sizeof(unsigned), stream);
  const unsigned totLen = iWidth * batchSize;
  if (totLen & 0xFF000000) {
    // BS: 1024 = 32 warps
    const unsigned BS = 1024;
    const unsigned GS = max(1024, (batchSize + 31) >> 5);
    batchedFilterNCompressKernel<KeyType, BS>
        <<<GS, BS, 0, stream>>>(batchSize, iWidth, oWidth, k, inKey, inThresh,
                                outKey, outIdx, outNumLeft);
  } else if (totLen & 0x00C00000) {
    // BS: 1024 = 32 warps
    const unsigned BS = 1024;
    const unsigned GS = max(512, (batchSize + 31) >> 5);
    batchedFilterNCompressKernel<KeyType, BS>
        <<<GS, BS, 0, stream>>>(batchSize, iWidth, oWidth, k, inKey, inThresh,
                                outKey, outIdx, outNumLeft);
  } else if (totLen & 0x00300000) {
    // BS: 256 = 8 warps
    const unsigned BS = 256;
    const unsigned GS = max(32, (batchSize + 7) >> 3);
    batchedFilterNCompressKernel<KeyType, BS>
        <<<GS, BS, 0, stream>>>(batchSize, iWidth, oWidth, k, inKey, inThresh,
                                outKey, outIdx, outNumLeft);
  } else if (totLen & 0x000FFFFC) {
    // BS: 128 = 4 warps
    const unsigned BS = 128;
    unsigned GS = min(((totLen >> 2) + 127) / 128, 128);
    GS = max(GS, (batchSize + 3) >> 2);
    batchedFilterNCompressKernel<KeyType, BS>
        <<<GS, BS, 0, stream>>>(batchSize, iWidth, oWidth, k, inKey, inThresh,
                                outKey, outIdx, outNumLeft);
  } else {
    const unsigned BS = 32;
    const unsigned GS = batchSize;
    batchedFilterNCompressKernel<KeyType, BS>
        <<<GS, BS, 0, stream>>>(batchSize, iWidth, oWidth, k, inKey, inThresh,
                                outKey, outIdx, outNumLeft);
  }
}

#endif  // DYN_TOP_K_FILTER_K_COMPRESS_H_

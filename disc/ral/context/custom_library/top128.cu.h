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

#ifndef DYN_TOP_K_TOP_128_H_
#define DYN_TOP_K_TOP_128_H_

#include "topk_util.cu.h"

// process an array with one warp
// if needed, keys should be initialized outside of the function
template <typename KeyType>
__inline__ __device__ void bitonicTop128(const unsigned iLen, KeyType keys[8],
                                         unsigned idx[8], KeyType* const inKey,
                                         const unsigned mask = full_mask) {
  const unsigned lane = threadIdx.x & 31;
  // one thread loading 4 elements at a time
  const unsigned roundLen = (iLen >> 5) << 5;
  unsigned i = lane, j = 0;
  int cond1 = i < roundLen;
  const unsigned mask1 = __ballot_sync(mask, cond1);
  for (; i < roundLen; i += 32, ++j) {
    keys[4 + (j & 3)] = inKey[i];
    idx[4 + (j & 3)] = i;
    int cond2 = ((j & 3) == 3);
    const unsigned mask2 = __ballot_sync(mask1, cond2);
    if (cond2) bitonicSort256_256(keys, idx, mask2);
  }
  if (i < iLen) {
    keys[4 + (j & 3)] = inKey[i];
    idx[4 + (j & 3)] = i;
  }
  int cond3 = (roundLen != iLen || (j & 3) != 0);
  const unsigned mask3 = __ballot_sync(mask, cond3);
  if (cond3) bitonicSort256_256(keys, idx, mask3);
}

// process an array with one warp
// if needed, keys should be initialized outside of the function
template <typename KeyType, typename ValType>
__inline__ __device__ void bitonicTop128(const unsigned iLen, KeyType keys[8],
                                         ValType vals[8], KeyType* const inKey,
                                         ValType* const inVal,
                                         const unsigned mask = full_mask) {
  const unsigned lane = threadIdx.x & 31;
  // one thread loading 4 elements at a time
  const unsigned roundLen = (iLen >> 5) << 5;
  unsigned i = lane, j = 0;
  int cond1 = i < roundLen;
  const unsigned mask1 = __ballot_sync(mask, cond1);
  for (; i < roundLen; i += 32, ++j) {
    keys[4 + (j & 3)] = inKey[i];
    vals[4 + (j & 3)] = inVal[i];
    int cond2 = ((j & 3) == 3);
    const unsigned mask2 = __ballot_sync(mask1, cond2);
    if (cond2) bitonicSort256_256(keys, vals, mask2);
  }
  if (i < iLen) {
    keys[4 + (j & 3)] = inKey[i];
    vals[4 + (j & 3)] = inVal[i];
  }
  int cond3 = (roundLen != iLen || (j & 3) != 0);
  const unsigned mask3 = __ballot_sync(mask, cond3);
  if (cond3) bitonicSort256_256(keys, vals, mask3);
}

template <typename KeyType, unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA) __global__
    void batchedTop128Kernel(const unsigned batchSize, const unsigned iWidth,
                             const unsigned oWidth, KeyType* inKey,
                             KeyType* outKey, unsigned* outIdx) {
  const unsigned batch = (blockIdx.x * nthdsPerCTA + threadIdx.x) >> 5;

  if (batch < batchSize) {
    KeyType keyBuffer[8];
    unsigned idxBuffer[8];
    keyBuffer[0] = keyBuffer[1] = keyBuffer[2] = keyBuffer[3] = keyBuffer[4] =
        keyBuffer[5] = keyBuffer[6] = keyBuffer[7] =
            std::numeric_limits<KeyType>::lowest();

    KeyType* const inKeyOff = inKey + batch * iWidth;
    bitonicTop128(iWidth, keyBuffer, idxBuffer, inKeyOff);

    // assumed at most top128
    KeyType* const outKeyOff = outKey + batch * oWidth;
    unsigned* const outIdxOff = outIdx + batch * oWidth;
    for (unsigned i = (threadIdx.x & 31), j = 0; i < oWidth && j < 4;
         i += 32, ++j) {
      outKeyOff[i] = keyBuffer[j];
      outIdxOff[i] = idxBuffer[j];
    }
  }
}

template <typename KeyType>
void batchedTop128Gpu(cudaStream_t stream, const unsigned batchSize,
                      const unsigned iWidth, const unsigned oWidth,
                      KeyType* inKey, KeyType* outKey, unsigned* outIdx) {
  // BS: 128 = 4 warps
  const unsigned BS = 128;
  const unsigned GS = (batchSize + 3) >> 2;
  batchedTop128Kernel<KeyType, BS>
      <<<GS, BS, 0, stream>>>(batchSize, iWidth, oWidth, inKey, outKey, outIdx);
}

void batchedTop128(cudaStream_t stream, const unsigned batchSize,
                   const unsigned iWidth, const unsigned oWidth, float* inKey,
                   float* outKey, unsigned* outIdx) {
  batchedTop128Gpu(stream, batchSize, iWidth, oWidth, inKey, outKey, outIdx);
}

template <typename KeyType, unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA) __global__
    void batchedFilteredTop128Kernel(const unsigned batchSize,
                                     const unsigned iLen, const unsigned oLen,
                                     const unsigned filtBufLen, KeyType* inKey,
                                     KeyType* filteredKey,
                                     unsigned* filteredIdx, KeyType* multiTop2,
                                     KeyType* thresholds, unsigned* numFiltered,
                                     KeyType* outKey, unsigned* outIdx) {
  __shared__ KeyType sKey[nthdsPerCTA >> 5][128];
  __shared__ unsigned sIdx[nthdsPerCTA >> 5][128];

  const unsigned batch = (blockIdx.x * nthdsPerCTA + threadIdx.x) >> 5;

  if (batch < batchSize) {
    KeyType keyBuffer[8];
    unsigned idxBuffer[8];
    keyBuffer[0] = keyBuffer[1] = keyBuffer[2] = keyBuffer[3] = keyBuffer[4] =
        keyBuffer[5] = keyBuffer[6] = keyBuffer[7] =
            std::numeric_limits<KeyType>::lowest();

    const unsigned lane = threadIdx.x & 31;

    int cond1 = (numFiltered[batch] > filtBufLen);
    unsigned mask1 = __ballot_sync(full_mask, cond1);

    if (!cond1) {
      KeyType* const keyOff = filteredKey + batch * filtBufLen;
      unsigned* const idxOff = filteredIdx + batch * filtBufLen;
      bitonicTop128(numFiltered[batch], keyBuffer, idxBuffer, keyOff, idxOff,
                    ~mask1);
    } else  // cond1 matched
    {
      const unsigned multiTop2Stride = oLen << 1;
      const KeyType* inKeyOff = inKey + batch * iLen;
      const KeyType thresh = thresholds[batch];

      const unsigned remainder = iLen % oLen;
      const unsigned roundUp = iceil(iLen, oLen);
      const unsigned roundDw = iLen / oLen;

      KeyType* locSKey = sKey[threadIdx.x >> 5];
      unsigned* locSIdx = sIdx[threadIdx.x >> 5];
      unsigned smemOff = 0;

      for (unsigned setId = 0; setId < oLen; ++setId) {
        int cond2 = (multiTop2[batch * multiTop2Stride + setId * 2] >= thresh);
        const unsigned mask2 = __ballot_sync(mask1, cond2);
        if (cond2) {
          const unsigned beg =
              setId < remainder ? roundUp * setId : roundDw * setId + remainder;
          const unsigned setLen = setId < remainder ? roundUp : roundDw;
          unsigned i = lane;

          int cond3 = (i < ((setLen >> 5) << 5));
          unsigned mask3 = __ballot_sync(mask2, cond3);
          while (i < ((setLen >> 5) << 5)) {
            unsigned idx = beg + i;
            KeyType key = inKeyOff[idx];
            unsigned len = warpFilterCompress(
                key, idx, thresh, &locSKey[smemOff], &locSIdx[smemOff], mask3);
            smemOff += len;
            int cond4 = (smemOff > 96);
            const unsigned mask4 = __ballot_sync(mask3, cond4);
            if (cond4) {
              bitonicTop128(smemOff, keyBuffer, idxBuffer, locSKey, locSIdx,
                            mask4);
              smemOff = 0;
            }
            i += 32;
            mask3 = __ballot_sync(mask3, (i < ((setLen >> 5) << 5)));
          }  // while:i

          unsigned len = 0;
          int cond5 = (i < setLen);
          const unsigned mask5 = __ballot_sync(mask2, cond5);
          if (cond5) {
            unsigned idx = beg + i;
            KeyType key = inKeyOff[idx];
            len = warpFilterCompress(key, idx, thresh, &locSKey[smemOff],
                                     &locSIdx[smemOff], mask5);
          }

          smemOff += __shfl_sync(mask2, len, 0);

          int cond6 = (smemOff > 96);
          const unsigned mask6 = __ballot_sync(mask2, cond6);
          if (cond6) {
            bitonicTop128(smemOff, keyBuffer, idxBuffer, locSKey, locSIdx,
                          mask6);
            smemOff = 0;
          }
        }  // if:multiTop2
      }    // for:setId
      for (unsigned i = lane, j = 0; i < smemOff && j < 4; i += 32, ++j) {
        keyBuffer[4 + j] = locSKey[i];
        idxBuffer[4 + j] = locSIdx[i];
      }
      int cond7 = (smemOff != 0);
      const unsigned mask7 = __ballot_sync(mask1, cond7);
      if (cond7) bitonicSort256_256(keyBuffer, idxBuffer, mask7);
    }  // cond1 matched
    // assumed at most top128
    KeyType* const outKeyOff = outKey + batch * oLen;
    unsigned* const outIdxOff = outIdx + batch * oLen;
    for (unsigned i = lane, j = 0; i < oLen && j < 4; i += 32, ++j) {
      outKeyOff[i] = keyBuffer[j];
      outIdxOff[i] = idxBuffer[j];
    }
  }  // if:batch
}

template <typename KeyType>
void batchedFilteredTop128Gpu(cudaStream_t stream, const unsigned batchSize,
                              const unsigned iLen, const unsigned oLen,
                              const unsigned filtBufLen, KeyType* inKey,
                              KeyType* filteredKey, unsigned* filteredIdx,
                              KeyType* multiTop2, KeyType* threshold,
                              unsigned* numFiltered, KeyType* outKey,
                              unsigned* outIdx) {
  const unsigned BS = 32;
  const unsigned GS = batchSize;
  batchedFilteredTop128Kernel<KeyType, BS><<<GS, BS, 0, stream>>>(
      batchSize, iLen, oLen, filtBufLen, inKey, filteredKey, filteredIdx,
      multiTop2, threshold, numFiltered, outKey, outIdx);
}

#endif  // DYN_TOP_K_TOP_128_H_

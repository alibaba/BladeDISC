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

#ifndef DYN_TOP_K_TOP_2_H_
#define DYN_TOP_K_TOP_2_H_

#include "topk_util.cu.h"

// this kernel function assumes a set is handled by at least one warp
// nthdsPerSet is assumed to be a multiple of 32
// k*2 is assumed to be smaller than width
template <typename Dtype, unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA) __global__
    void batchedMultiSetsTop2Kernel(const unsigned batchSize,
                                    const unsigned iWidth,
                                    const unsigned oWidth, const unsigned k,
                                    Dtype* iKey, Dtype* oKey) {
  const unsigned gid = blockIdx.x * nthdsPerCTA + threadIdx.x;
  const unsigned nWarps = (nthdsPerCTA * gridDim.x) >> 5;
  const unsigned wid = gid >> 5;

  const unsigned nSets = k * batchSize;
  unsigned remainder = nWarps % nSets;
  unsigned roundup = iceil(nWarps, nSets);
  unsigned rounddw = nWarps / nSets;
  // warpSetID is the ID of a set that the current warp needs to handle
  const unsigned warpSetID =
      (wid / roundup) < remainder ? wid / roundup : (wid - remainder) / rounddw;
  const unsigned setInitWarp = warpSetID < remainder
                                   ? roundup * warpSetID
                                   : rounddw * warpSetID + remainder;
  const unsigned setNumThreads =
      warpSetID < remainder ? (roundup << 5) : (rounddw << 5);

  remainder = iWidth % k;
  roundup = iceil(iWidth, k);
  rounddw = iWidth / k;

  const unsigned pos = warpSetID / k;
  // batchSetID is the ID of the set within the current batch
  const unsigned batchSetID = warpSetID % k;

  const unsigned beg = batchSetID < remainder
                           ? roundup * batchSetID
                           : rounddw * batchSetID + remainder;
  const unsigned len = batchSetID < remainder ? roundup : rounddw;

  Dtype key1 = std::numeric_limits<Dtype>::lowest();
  Dtype key2 = std::numeric_limits<Dtype>::lowest();

  for (unsigned i = gid - (setInitWarp << 5); i < len; i += setNumThreads) {
    Dtype key = (iKey + pos * iWidth + beg)[i];
    if (key1 < key) {
      key2 = key1;
      key1 = key;
    } else if (key2 < key) {
      key2 = key;
    }
  }

  Dtype locKey1, locKey2;
  warpReduceTop2(key1, &locKey1, &locKey2);
  key2 = warpReduceMax(key2);

  if ((threadIdx.x & 31) == 0) {
    locKey2 = max(locKey2, key2);
    Dtype oldKey = atomicMax(&oKey[pos * oWidth + batchSetID * 2], locKey1);
    if (locKey1 > oldKey) {
      locKey2 = max(oldKey, locKey2);
      atomicMax(&oKey[pos * oWidth + batchSetID * 2 + 1], locKey2);
    } else {
      atomicMax(&oKey[pos * oWidth + batchSetID * 2 + 1], locKey1);
    }
  }
}

template <typename Dtype>
void batchedMultiSetsTop2Gpu(cudaStream_t stream, const unsigned batchSize,
                             const unsigned iWidth, const unsigned oWidth,
                             const unsigned k, Dtype* iKey, Dtype* oKey) {
  batchedSetToLowest<Dtype, 128>
      <<<batchSize, 128, 0, stream>>>(batchSize, 2 * k, oWidth, oKey);
  const unsigned totLen = iWidth * batchSize;
  const unsigned nSets = k * batchSize;
  if (totLen & 0xFF000000) {
    // BS: 1024 = 32 warps
    const unsigned BS = 1024;
    const unsigned GS = max(1024, (nSets + 31) >> 5);
    batchedMultiSetsTop2Kernel<Dtype, BS>
        <<<GS, BS, 0, stream>>>(batchSize, iWidth, oWidth, k, iKey, oKey);
  } else if (totLen & 0x00C00000) {
    // BS: 1024 = 32 warps
    const unsigned BS = 1024;
    const unsigned GS = max(512, (nSets + 31) >> 5);
    batchedMultiSetsTop2Kernel<Dtype, BS>
        <<<GS, BS, 0, stream>>>(batchSize, iWidth, oWidth, k, iKey, oKey);
  } else if (totLen & 0x00300000) {
    // BS: 256 = 8 warps
    const unsigned BS = 256;
    const unsigned GS = max(32, (nSets + 7) >> 3);
    batchedMultiSetsTop2Kernel<Dtype, BS>
        <<<GS, BS, 0, stream>>>(batchSize, iWidth, oWidth, k, iKey, oKey);
  } else if (totLen & 0x000FFFFC) {
    // BS: 128 = 4 warps
    const unsigned BS = 128;
    const unsigned GS = max(128, (nSets + 3) >> 2);
    batchedMultiSetsTop2Kernel<Dtype, BS>
        <<<GS, BS, 0, stream>>>(batchSize, iWidth, oWidth, k, iKey, oKey);
  } else {
    const unsigned BS = 32;
    const unsigned GS = nSets;
    batchedMultiSetsTop2Kernel<Dtype, BS>
        <<<GS, BS, 0, stream>>>(batchSize, iWidth, oWidth, k, iKey, oKey);
  }
}

#endif  // DYN_TOP_K_TOP_2_H_

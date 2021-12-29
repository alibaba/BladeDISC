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

#ifndef DYN_TOP_K_SMALL_TOP_128_H_
#define DYN_TOP_K_SMALL_TOP_128_H_

#include "topk_util.cu.h"

template <typename Dtype>
struct vecType;

template <>
struct vecType<unsigned int> {
  typedef uint2 vec2;
  typedef uint4 vec4;
};

template <>
struct vecType<int> {
  typedef int2 vec2;
  typedef int4 vec4;
};

template <>
struct vecType<float> {
  typedef float2 vec2;
  typedef float4 vec4;
};

// this device function is only suitable for one active warp
template <typename Itype, typename Dtype>
__inline__ __device__ void vec4LoadAndBitonicSort128(Dtype* const iKey,
                                                     Dtype keys[8],
                                                     Itype vals[8],
                                                     const Itype iLen) {
  typedef typename vecType<Dtype>::vec4 Dvec4;

  keys[0] = std::numeric_limits<Dtype>::lowest();
  keys[1] = std::numeric_limits<Dtype>::lowest();
  keys[2] = std::numeric_limits<Dtype>::lowest();
  keys[3] = std::numeric_limits<Dtype>::lowest();
  keys[4] = std::numeric_limits<Dtype>::lowest();
  keys[5] = std::numeric_limits<Dtype>::lowest();
  keys[6] = std::numeric_limits<Dtype>::lowest();
  keys[7] = std::numeric_limits<Dtype>::lowest();

  const unsigned lane = threadIdx.x & 31;
  const unsigned offsetBytes = (uintptr_t)iKey & ((sizeof(Dtype) << 2) - 1);
  const Itype peel =
      (offsetBytes == 0 ? 0 : (sizeof(Dtype) << 2) - offsetBytes) /
      sizeof(Dtype);
  Itype i = lane;
  if (i < min(peel, iLen)) {
    keys[0] = iKey[i];
    vals[0] = i;
  }

  // must read 128 elements at a time
  const Itype nVecLd = iLen < peel ? 0 : (iLen - peel) >> 7;
  const Itype roundLen = nVecLd << 5;
  for (; i < roundLen; i += 32) {
    Dvec4 key = reinterpret_cast<Dvec4*>(iKey + peel)[i];

    keys[4] = key.x;
    keys[5] = key.y;
    keys[6] = key.z;
    keys[7] = key.w;

    vals[4] = peel + (i << 2);
    vals[5] = peel + (i << 2) + 1;
    vals[6] = peel + (i << 2) + 2;
    vals[7] = peel + (i << 2) + 3;

    int cond1 = (i < roundLen);
    const unsigned mask1 = __ballot_sync(full_mask, cond1);
    bitonicSort256_256(keys, vals, mask1);
  }
  i = peel + (roundLen << 2) + lane;
  if (i < iLen) {
    keys[4] = iKey[i];
    vals[4] = i;
    i += 32;
  }
  if (i < iLen) {
    keys[5] = iKey[i];
    vals[5] = i;
    i += 32;
  }
  if (i < iLen) {
    keys[6] = iKey[i];
    vals[6] = i;
    i += 32;
  }
  if (i < iLen) {
    keys[7] = iKey[i];
    vals[7] = i;
    i += 32;
  }
  bitonicSort256_256(keys, vals, full_mask);
}

// this kernel is designed for topk where k<=128
// this kernel supports in-place sorting (meaning ikey could be the same as
// okey) gridDim.x is assumed to be equal to batch the value is the local index
// (index within a batch)
template <typename Itype, typename Dtype, unsigned nthds_per_block>
__launch_bounds__(nthds_per_block) __global__
    void batchTop128(Dtype* ikey, Dtype* okey, Itype* oval, Itype batch,
                     Itype iwidth, Itype owidth) {
  const Itype pos = (blockIdx.x * nthds_per_block + threadIdx.x) >> 5;

  if (pos < batch) {
    // Obtain a segment of consecutive items that are blocked across threads
    Dtype thread_keys[8];
    Itype thread_vals[8];

    Dtype* const iKeyOfs = ikey + pos * iwidth;
    vec4LoadAndBitonicSort128(iKeyOfs, thread_keys, thread_vals, iwidth);

    // assumed at most 128 top elements
    Itype i = threadIdx.x & 31;
    Dtype* const okeyOfs = okey + pos * owidth;
    Itype* const ovalOfs = oval + pos * owidth;
    if (i < owidth) {
      okeyOfs[i] = thread_keys[0];
      ovalOfs[i] = thread_vals[0];
      i += 32;
    }
    if (i < owidth) {
      okeyOfs[i] = thread_keys[1];
      ovalOfs[i] = thread_vals[1];
      i += 32;
    }
    if (i < owidth) {
      okeyOfs[i] = thread_keys[2];
      ovalOfs[i] = thread_vals[2];
      i += 32;
    }
    if (i < owidth) {
      okeyOfs[i] = thread_keys[3];
      ovalOfs[i] = thread_vals[3];
      i += 32;
    }
  }  // if:pos<batch
}

template <typename Itype, typename Dtype>
void batchTop128Launch(Dtype* ikey, Dtype* okey, Itype* oval, Itype batch,
                       Itype iwidth, Itype owidth, cudaStream_t stream = NULL) {
  const unsigned blocks = (batch + 3) / 4;
  batchTop128<Itype, Dtype, 128>
      <<<blocks, 128, 0, stream>>>(ikey, okey, oval, batch, iwidth, owidth);
}

#endif  // DYN_TOP_K_SMALL_TOP_128_H_

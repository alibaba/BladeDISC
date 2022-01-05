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

#ifndef DYN_SORT_WARP_H_
#define DYN_SORT_WARP_H_

#include <stdio.h>

#include <limits>

template <typename Dtype>
__inline__ __device__ Dtype warpReduceSum(Dtype val,
                                          const unsigned mask = full_mask) {
  val += __shfl_down_sync(mask, val, 16);
  val += __shfl_down_sync(mask, val, 8);
  val += __shfl_down_sync(mask, val, 4);
  val += __shfl_down_sync(mask, val, 2);
  val += __shfl_down_sync(mask, val, 1);
  return val;
}

template <typename Dtype>
__inline__ __device__ Dtype warpReduceMax(Dtype val,
                                          const unsigned mask = full_mask) {
  Dtype tmp = __shfl_down_sync(mask, val, 16);
  if (val < tmp) val = tmp;
  tmp = __shfl_down_sync(mask, val, 8);
  if (val < tmp) val = tmp;
  tmp = __shfl_down_sync(mask, val, 4);
  if (val < tmp) val = tmp;
  tmp = __shfl_down_sync(mask, val, 2);
  if (val < tmp) val = tmp;
  tmp = __shfl_down_sync(mask, val, 1);
  if (val < tmp) val = tmp;
  return val;
}

__inline__ __device__ int warpReduceMax_alternative(
    int val, const unsigned mask = full_mask) {
  int tmp = __shfl_down_sync(mask, val, 16);
  val = max(val, tmp);
  tmp = __shfl_down_sync(mask, val, 8);
  val = max(val, tmp);
  tmp = __shfl_down_sync(mask, val, 4);
  val = max(val, tmp);
  tmp = __shfl_down_sync(mask, val, 2);
  val = max(val, tmp);
  tmp = __shfl_down_sync(mask, val, 1);
  val = max(val, tmp);
  return val;
}

template <typename Dtype, typename Itype>
__inline__ __device__ void warpReduceMaxWithPos(
    Dtype val, Itype pos, Dtype* maxval, Itype* maxpos,
    const unsigned mask = full_mask) {
  *maxval = val;
  *maxpos = pos;
  // if the thread that should give the value is inactive
  // the data read is unpredictable
  Dtype tmp1 = __shfl_down_sync(mask, *maxval, 16);
  Dtype tmp2 = __shfl_down_sync(mask, *maxpos, 16);
  if (*maxval < tmp1) {
    *maxval = tmp1;
    *maxpos = tmp2;
  }
  tmp1 = __shfl_down_sync(mask, *maxval, 8);
  tmp2 = __shfl_down_sync(mask, *maxpos, 8);
  if (*maxval < tmp1) {
    *maxval = tmp1;
    *maxpos = tmp2;
  }
  tmp1 = __shfl_down_sync(mask, *maxval, 4);
  tmp2 = __shfl_down_sync(mask, *maxpos, 4);
  if (*maxval < tmp1) {
    *maxval = tmp1;
    *maxpos = tmp2;
  }
  tmp1 = __shfl_down_sync(mask, *maxval, 2);
  tmp2 = __shfl_down_sync(mask, *maxpos, 2);
  if (*maxval < tmp1) {
    *maxval = tmp1;
    *maxpos = tmp2;
  }
  tmp1 = __shfl_down_sync(mask, *maxval, 1);
  tmp2 = __shfl_down_sync(mask, *maxpos, 1);
  if (*maxval < tmp1) {
    *maxval = tmp1;
    *maxpos = tmp2;
  }
}

template <typename Dtype>
// &val and fst should not be the same
__inline__ __device__ void warpReduceTop2(Dtype val, Dtype* fst, Dtype* snd,
                                          const unsigned mask = full_mask) {
  // if the thread that should give the value is inactive
  // the data read is unpredictable
  unsigned laneid = threadIdx.x & 31;
  unsigned pos;
  warpReduceMaxWithPos(val, laneid, fst, &pos);
  pos = __shfl_sync(mask, pos, 0);
  if (laneid == pos) val = std::numeric_limits<Dtype>::lowest();
  *snd = warpReduceMax(val);
}

// &val and fst should not be the same
template <typename Itype, typename Dtype>
__inline__ __device__ void warpReduceTop2WithPos(
    Dtype val, Itype pos, Dtype* fstval, Itype* fstpos, Dtype* sndval,
    Itype* sndpos, const unsigned mask = full_mask) {
  // if the thread that should give the value is inactive
  // the data read is unpredictable
  warpReduceMaxWithPos(val, pos, fstval, fstpos);
  if (pos == __shfl_sync(mask, *fstpos, 0))
    val = std::numeric_limits<Dtype>::lowest();
  warpReduceMaxWithPos(val, pos, sndval, sndpos);
}

__inline__ __device__ int warpScan(int val, const unsigned mask = full_mask) {
  int lane = threadIdx.x & 31;
  int tmp = __shfl_up_sync(mask, val, 1);
  if (lane > 0) val += tmp;
  tmp = __shfl_up_sync(mask, val, 2);
  if (lane > 1) val += tmp;
  tmp = __shfl_up_sync(mask, val, 4);
  if (lane > 3) val += tmp;
  tmp = __shfl_up_sync(mask, val, 8);
  if (lane > 7) val += tmp;
  tmp = __shfl_up_sync(mask, val, 16);
  if (lane > 15) val += tmp;
  return val;
}

template <typename Itype, typename Dtype>
__device__ __inline__ int warpFilterCompress(Dtype key, Itype val,
                                             Dtype threshold, Dtype* keyBuffer,
                                             Itype* valBuffer,
                                             const unsigned mask = full_mask) {
  unsigned int valid = key >= threshold ? 1 : 0;
  int len = __popc(__ballot_sync(mask, valid));
  if (len == 0) return 0;
  int wt_ofs = warpScan(valid, mask) - 1;
  if (valid) {
    keyBuffer[wt_ofs] = key;
    valBuffer[wt_ofs] = val;
  }
  return len;
}

#endif  // DYN_SORT_WARP_H_

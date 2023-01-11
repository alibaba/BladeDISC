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

#ifndef DYN_SORT_BITONIC_SORT_H_
#define DYN_SORT_BITONIC_SORT_H_

template <typename Dtype>
__device__ __forceinline__ void bitonicExchange(
    Dtype& key, const unsigned mask, const unsigned active_mask = full_mask) {
  const Dtype key1 = key;
  const unsigned tgx = threadIdx.x & 31;
  const unsigned otgx = tgx ^ mask;
  const Dtype key2 = __shfl_sync(active_mask, key, otgx);
  const bool flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2);
  key = flag ? key1 : key2;
}

template <typename Itype, typename Dtype>
__device__ __forceinline__ void bitonicExchangePair(
    Dtype& key, Itype& val, const unsigned mask,
    const unsigned active_mask = full_mask) {
  const Dtype key1 = key;
  const Itype val1 = val;
  const unsigned tgx = threadIdx.x & 31;
  const unsigned otgx = tgx ^ mask;
  const Dtype key2 = __shfl_sync(active_mask, key, otgx);
  const Dtype val2 = __shfl_sync(active_mask, val, otgx);
  const bool flag = ((key1 > key2) ^ (tgx > otgx)) && (key1 != key2);
  key = flag ? key1 : key2;
  val = flag ? val1 : val2;
}

// warp bitonic sort
template <typename Dtype>
__device__ __inline__ void bitonicSort32(
    Dtype& key, const unsigned active_mask = full_mask) {
  bitonicExchange(key, 1, active_mask);
  bitonicExchange(key, 3, active_mask);
  bitonicExchange(key, 1, active_mask);
  bitonicExchange(key, 7, active_mask);
  bitonicExchange(key, 2, active_mask);
  bitonicExchange(key, 1, active_mask);
  bitonicExchange(key, 15, active_mask);
  bitonicExchange(key, 4, active_mask);
  bitonicExchange(key, 2, active_mask);
  bitonicExchange(key, 1, active_mask);
  bitonicExchange(key, 31, active_mask);
  bitonicExchange(key, 8, active_mask);
  bitonicExchange(key, 4, active_mask);
  bitonicExchange(key, 2, active_mask);
  bitonicExchange(key, 1, active_mask);
}

// warp bitonic sort
template <typename Itype, typename Dtype>
__device__ __inline__ void bitonicSort32(
    Dtype& key, Itype& val, const unsigned active_mask = full_mask) {
  bitonicExchangePair(key, val, 1, active_mask);
  bitonicExchangePair(key, val, 3, active_mask);
  bitonicExchangePair(key, val, 1, active_mask);
  bitonicExchangePair(key, val, 7, active_mask);
  bitonicExchangePair(key, val, 2, active_mask);
  bitonicExchangePair(key, val, 1, active_mask);
  bitonicExchangePair(key, val, 15, active_mask);
  bitonicExchangePair(key, val, 4, active_mask);
  bitonicExchangePair(key, val, 2, active_mask);
  bitonicExchangePair(key, val, 1, active_mask);
  bitonicExchangePair(key, val, 31, active_mask);
  bitonicExchangePair(key, val, 8, active_mask);
  bitonicExchangePair(key, val, 4, active_mask);
  bitonicExchangePair(key, val, 2, active_mask);
  bitonicExchangePair(key, val, 1, active_mask);
}

template <typename Itype, typename Dtype>
__device__ __inline__ void bitonicExchange256Pair(
    Dtype keys[8], Itype vals[8], const unsigned mask,
    const unsigned active_mask = full_mask) {
  bitonicExchangePair(keys[0], vals[0], mask, active_mask);
  bitonicExchangePair(keys[1], vals[1], mask, active_mask);
  bitonicExchangePair(keys[2], vals[2], mask, active_mask);
  bitonicExchangePair(keys[3], vals[3], mask, active_mask);
  bitonicExchangePair(keys[4], vals[4], mask, active_mask);
  bitonicExchangePair(keys[5], vals[5], mask, active_mask);
  bitonicExchangePair(keys[6], vals[6], mask, active_mask);
  bitonicExchangePair(keys[7], vals[7], mask, active_mask);
}

template <typename Dtype>
__device__ __inline__ void bitonicExchange256(
    Dtype keys[8], const unsigned mask,
    const unsigned active_mask = full_mask) {
  bitonicExchange(keys[0], mask, active_mask);
  bitonicExchange(keys[1], mask, active_mask);
  bitonicExchange(keys[2], mask, active_mask);
  bitonicExchange(keys[3], mask, active_mask);
  bitonicExchange(keys[4], mask, active_mask);
  bitonicExchange(keys[5], mask, active_mask);
  bitonicExchange(keys[6], mask, active_mask);
  bitonicExchange(keys[7], mask, active_mask);
}

template <typename Itype, typename Dtype>
__device__ __inline__ void bitonicSort32_256_pair(
    Dtype keys[8], Itype vals[8], const unsigned active_mask = full_mask) {
  bitonicExchange256Pair(keys, vals, 1, active_mask);
  bitonicExchange256Pair(keys, vals, 3, active_mask);
  bitonicExchange256Pair(keys, vals, 1, active_mask);
  bitonicExchange256Pair(keys, vals, 7, active_mask);
  bitonicExchange256Pair(keys, vals, 2, active_mask);
  bitonicExchange256Pair(keys, vals, 1, active_mask);
  bitonicExchange256Pair(keys, vals, 15, active_mask);
  bitonicExchange256Pair(keys, vals, 4, active_mask);
  bitonicExchange256Pair(keys, vals, 2, active_mask);
  bitonicExchange256Pair(keys, vals, 1, active_mask);
  bitonicExchange256Pair(keys, vals, 31, active_mask);
  bitonicExchange256Pair(keys, vals, 8, active_mask);
  bitonicExchange256Pair(keys, vals, 4, active_mask);
  bitonicExchange256Pair(keys, vals, 2, active_mask);
  bitonicExchange256Pair(keys, vals, 1, active_mask);
}

template <typename Dtype>
__device__ __inline__ void bitonicSort32_256(
    Dtype keys[8], const unsigned active_mask = full_mask) {
  bitonicExchange256(keys, 1, active_mask);
  bitonicExchange256(keys, 3, active_mask);
  bitonicExchange256(keys, 1, active_mask);
  bitonicExchange256(keys, 7, active_mask);
  bitonicExchange256(keys, 2, active_mask);
  bitonicExchange256(keys, 1, active_mask);
  bitonicExchange256(keys, 15, active_mask);
  bitonicExchange256(keys, 4, active_mask);
  bitonicExchange256(keys, 2, active_mask);
  bitonicExchange256(keys, 1, active_mask);
  bitonicExchange256(keys, 31, active_mask);
  bitonicExchange256(keys, 8, active_mask);
  bitonicExchange256(keys, 4, active_mask);
  bitonicExchange256(keys, 2, active_mask);
  bitonicExchange256(keys, 1, active_mask);
}

template <typename Itype, typename Dtype>
__device__ __forceinline__ void bitonicMergePair(
    Dtype& k0, Itype& v0, Dtype& k1, Itype& v1,
    const unsigned active_mask = full_mask) {
  const unsigned tgx = threadIdx.x & 31;
  const unsigned otgx = 31 - tgx;
  Dtype key1 = k0;
  Dtype val1 = v0;
  const Dtype key2 = __shfl_sync(active_mask, k1, otgx);
  const Dtype val2 = __shfl_sync(active_mask, v1, otgx);
  const bool flag = key1 > key2;
  k0 = flag ? key1 : key2;
  v0 = flag ? val1 : val2;
  key1 = flag ? key2 : key1;
  val1 = flag ? val2 : val1;
  k1 = __shfl_sync(active_mask, key1, otgx);
  v1 = __shfl_sync(active_mask, val1, otgx);
}

template <typename Dtype>
__device__ __forceinline__ void bitonicMerge(
    Dtype& k0, Dtype& k1, const unsigned active_mask = full_mask) {
  const unsigned tgx = threadIdx.x & 31;
  const unsigned otgx = 31 - tgx;
  Dtype key1 = k0;
  const Dtype key2 = __shfl_sync(active_mask, k1, otgx);
  const bool flag = key1 > key2;
  k0 = flag ? key1 : key2;
  key1 = flag ? key2 : key1;
  k1 = __shfl_sync(active_mask, key1, otgx);
}

template <typename Itype, typename Dtype>
__device__ __inline__ void bitonicMerge64_256_pair(
    Dtype keys[8], Itype vals[8], const unsigned active_mask = full_mask) {
  bitonicMergePair(keys[0], vals[0], keys[1], vals[1], active_mask);
  bitonicMergePair(keys[2], vals[2], keys[3], vals[3], active_mask);
  bitonicMergePair(keys[4], vals[4], keys[5], vals[5], active_mask);
  bitonicMergePair(keys[6], vals[6], keys[7], vals[7], active_mask);
}

template <typename Dtype>
__device__ __inline__ void bitonicMerge64_256(
    Dtype keys[8], const unsigned active_mask = full_mask) {
  bitonicMerge(keys[0], keys[1], active_mask);
  bitonicMerge(keys[2], keys[3], active_mask);
  bitonicMerge(keys[4], keys[5], active_mask);
  bitonicMerge(keys[6], keys[7], active_mask);
}

template <typename Itype, typename Dtype>
__device__ __inline__ void bitonicSort64_256_pair(
    Dtype keys[8], Itype vals[8], const unsigned active_mask = full_mask) {
  bitonicSort32_256_pair(keys, vals, active_mask);
  bitonicMerge64_256_pair(keys, vals, active_mask);
  bitonicExchange256Pair(keys, vals, 16, active_mask);
  bitonicExchange256Pair(keys, vals, 8, active_mask);
  bitonicExchange256Pair(keys, vals, 4, active_mask);
  bitonicExchange256Pair(keys, vals, 2, active_mask);
  bitonicExchange256Pair(keys, vals, 1, active_mask);
}

template <typename Dtype>
__device__ __inline__ void bitonicSort64_256(
    Dtype keys[8], const unsigned active_mask = full_mask) {
  bitonicSort32_256(keys, active_mask);
  bitonicMerge64_256(keys, active_mask);
  bitonicExchange256(keys, 16, active_mask);
  bitonicExchange256(keys, 8, active_mask);
  bitonicExchange256(keys, 4, active_mask);
  bitonicExchange256(keys, 2, active_mask);
  bitonicExchange256(keys, 1, active_mask);
}

template <typename Itype, typename Dtype>
__device__ __inline__ void bitonicMerge128_256_pair(
    Dtype keys[8], Itype vals[8], const unsigned active_mask = full_mask) {
  bitonicMergePair(keys[0], vals[0], keys[3], vals[3], active_mask);
  bitonicMergePair(keys[1], vals[1], keys[2], vals[2], active_mask);
  bitonicMergePair(keys[4], vals[4], keys[7], vals[7], active_mask);
  bitonicMergePair(keys[5], vals[5], keys[6], vals[6], active_mask);
}

template <typename Dtype>
__device__ __inline__ void bitonicMerge128_256(
    Dtype keys[8], const unsigned active_mask = full_mask) {
  bitonicMerge(keys[0], keys[3], active_mask);
  bitonicMerge(keys[1], keys[2], active_mask);
  bitonicMerge(keys[4], keys[7], active_mask);
  bitonicMerge(keys[5], keys[6], active_mask);
}

template <typename Itype, typename Dtype>
__device__ __inline__ void bitonicMerge256_256_pair(
    Dtype keys[8], Itype vals[8], const unsigned active_mask = full_mask) {
  bitonicMergePair(keys[0], vals[0], keys[7], vals[7], active_mask);
  bitonicMergePair(keys[1], vals[1], keys[6], vals[6], active_mask);
  bitonicMergePair(keys[2], vals[2], keys[5], vals[5], active_mask);
  bitonicMergePair(keys[3], vals[3], keys[4], vals[4], active_mask);
}

template <typename Dtype>
__device__ __inline__ void bitonicMerge256_256(
    Dtype keys[8], const unsigned active_mask = full_mask) {
  bitonicMerge(keys[0], keys[7], active_mask);
  bitonicMerge(keys[1], keys[6], active_mask);
  bitonicMerge(keys[2], keys[5], active_mask);
  bitonicMerge(keys[3], keys[4], active_mask);
}

template <typename T>
__device__ __inline__ static void swap(T& a, T& b) {
  T c(a);
  a = b;
  b = c;
}

template <typename Itype, typename Dtype>
__device__ __forceinline__ void condExchange(Dtype& k0, Itype& v0, Dtype& k1,
                                             Itype& v1) {
  if (k0 < k1) {
    swap(k0, k1);
    swap(v0, v1);
  }
}

template <typename Dtype>
__device__ __forceinline__ void condExchange(Dtype& k0, Dtype& k1) {
  if (k0 < k1) {
    swap(k0, k1);
  }
}

template <typename Itype, typename Dtype>
__device__ __inline__ void bitonicExchange32_256_pair(Dtype keys[8],
                                                      Itype vals[8]) {
  condExchange(keys[0], vals[0], keys[1], vals[1]);
  condExchange(keys[2], vals[2], keys[3], vals[3]);
  condExchange(keys[4], vals[4], keys[5], vals[5]);
  condExchange(keys[6], vals[6], keys[7], vals[7]);
}

template <typename Dtype>
__device__ __inline__ void bitonicExchange32_256(Dtype keys[8]) {
  condExchange(keys[0], keys[1]);
  condExchange(keys[2], keys[3]);
  condExchange(keys[4], keys[5]);
  condExchange(keys[6], keys[7]);
}

template <typename Itype, typename Dtype>
__device__ __inline__ void bitonicExchange64_256_pair(Dtype keys[8],
                                                      Itype vals[8]) {
  condExchange(keys[0], vals[0], keys[2], vals[2]);
  condExchange(keys[1], vals[1], keys[3], vals[3]);
  condExchange(keys[4], vals[4], keys[6], vals[6]);
  condExchange(keys[5], vals[5], keys[7], vals[7]);
}

template <typename Dtype>
__device__ __inline__ void bitonicExchange64_256(Dtype keys[8]) {
  condExchange(keys[0], keys[2]);
  condExchange(keys[1], keys[3]);
  condExchange(keys[4], keys[6]);
  condExchange(keys[5], keys[7]);
}

template <typename Itype, typename Dtype>
__device__ __inline__ void bitonicSort128_256_pair(
    Dtype keys[8], Itype vals[8], const unsigned active_mask = full_mask) {
  bitonicSort64_256_pair(keys, vals, active_mask);
  bitonicMerge128_256_pair(keys, vals, active_mask);
  bitonicExchange32_256_pair(keys, vals);
  bitonicExchange256Pair(keys, vals, 16, active_mask);
  bitonicExchange256Pair(keys, vals, 8, active_mask);
  bitonicExchange256Pair(keys, vals, 4, active_mask);
  bitonicExchange256Pair(keys, vals, 2, active_mask);
  bitonicExchange256Pair(keys, vals, 1, active_mask);
}

template <typename Dtype>
__device__ __inline__ void bitonicSort128_256(
    Dtype keys[8], const unsigned active_mask = full_mask) {
  bitonicSort64_256(keys, active_mask);
  bitonicMerge128_256(keys, active_mask);
  bitonicExchange32_256(keys);
  bitonicExchange256(keys, 16, active_mask);
  bitonicExchange256(keys, 8, active_mask);
  bitonicExchange256(keys, 4, active_mask);
  bitonicExchange256(keys, 2, active_mask);
  bitonicExchange256(keys, 1, active_mask);
}

template <typename Itype, typename Dtype>
__device__ __inline__ void bitonicSort256_256(
    Dtype keys[8], Itype vals[8], const unsigned active_mask = full_mask) {
  bitonicSort128_256_pair(keys, vals, active_mask);
  bitonicMerge256_256_pair(keys, vals, active_mask);
  bitonicExchange64_256_pair(keys, vals);
  bitonicExchange32_256_pair(keys, vals);
  bitonicExchange256Pair(keys, vals, 16, active_mask);
  bitonicExchange256Pair(keys, vals, 8, active_mask);
  bitonicExchange256Pair(keys, vals, 4, active_mask);
  bitonicExchange256Pair(keys, vals, 2, active_mask);
  bitonicExchange256Pair(keys, vals, 1, active_mask);
}

template <typename Dtype>
__device__ __inline__ void bitonicSort256_256(
    Dtype keys[8], const unsigned active_mask = full_mask) {
  bitonicSort128_256(keys, active_mask);
  bitonicMerge256_256(keys, active_mask);
  bitonicExchange64_256(keys);
  bitonicExchange32_256(keys);
  bitonicExchange256(keys, 16, active_mask);
  bitonicExchange256(keys, 8, active_mask);
  bitonicExchange256(keys, 4, active_mask);
  bitonicExchange256(keys, 2, active_mask);
  bitonicExchange256(keys, 1, active_mask);
}

#endif  // DYN_SORT_BITONIC_SORT_H_

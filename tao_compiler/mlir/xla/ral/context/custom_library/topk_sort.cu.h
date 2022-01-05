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

#ifndef DYN_SORT_TOP_K_SORT_H_
#define DYN_SORT_TOP_K_SORT_H_

#include <stdio.h>

#include <cstdint>

#include "filter_n_compress.cu.h"
#include "kth_biggest.cu.h"
#include "small_top128.cu.h"
#include "top128.cu.h"
#include "top2.cu.h"

static const unsigned cuMemAlign = 256;
static const unsigned batchBufLenFactor = 2;

// ALIGNPTR
int8_t* alignPtr(int8_t* ptr, uintptr_t to) {
  uintptr_t addr = (uintptr_t)ptr;
  if (addr % to) {
    addr += to - addr % to;
  }
  return (int8_t*)addr;
}

// NEXTWORKSPACEPTR
int8_t* nextWorkspacePtr(int8_t* ptr, uintptr_t previousWorkspaceSize) {
  uintptr_t addr = (uintptr_t)ptr;
  addr += previousWorkspaceSize;
  return alignPtr((int8_t*)addr, cuMemAlign);
}

// CALCULATE TOTAL WORKSPACE SIZE
size_t calculateTotalWorkspaceSize(size_t* workspaces, int count) {
  size_t total = 0;
  for (int i = 0; i < count; i++) {
    total += workspaces[i];
    if (workspaces[i] % cuMemAlign) {
      total += cuMemAlign - (workspaces[i] % cuMemAlign);
    }
  }
  return total;
}

template <typename DType>
size_t resultBufferSize(const unsigned batchSize, const unsigned k) {
  return batchBufLenFactor * k * batchSize * sizeof(DType);
}

template <typename DType>
size_t statusBufferSize(const unsigned batchSize) {
  return batchSize * sizeof(DType);
}

template <typename Itype, typename Dtype>
void batchedTopK(Dtype* ikey, Itype* ival, const Itype iLen, Dtype* okey,
                 Itype* oval, const Itype oLen, void* tempBuffer,
                 size_t& worksize, const Itype batch, cudaStream_t stream = 0) {
  const Itype tempWidth = 2 * oLen;
  const Itype tempLen = batch * tempWidth;
  const Itype peel = 4 - ((tempLen + tempLen + batch) & 3);

  if (tempBuffer == nullptr) {
    if (iLen < 8192) {
      worksize = 4;
    } else {
      worksize = (tempLen + tempLen + batch + peel) * sizeof(Dtype) +
                 (tempLen + batch) * sizeof(Itype);
    }
    return;
  }

  if (iLen < 8192) {
    batchTop128Launch(ikey, okey, oval, batch, iLen, oLen, stream);
    return;
  }

  Dtype* keyBuffer = (Dtype*)tempBuffer;
  Dtype* multiTop2 = (Dtype*)nextWorkspacePtr(
      (int8_t*)keyBuffer, resultBufferSize<Dtype>(batch, oLen));
  Dtype* threshold = (Dtype*)nextWorkspacePtr(
      (int8_t*)multiTop2, resultBufferSize<Dtype>(batch, oLen));
  unsigned* idxBuffer = (unsigned*)nextWorkspacePtr(
      (int8_t*)threshold, statusBufferSize<Dtype>(batch));
  unsigned* numFiltered = (unsigned*)nextWorkspacePtr(
      (int8_t*)idxBuffer, statusBufferSize<unsigned>(batch));

  const unsigned batchBufLen = batchBufLenFactor * oLen;

  batchedMultiSetsTop2Gpu(stream, batch, iLen, batchBufLen, oLen, (Dtype*)ikey,
                          multiTop2);

  batchedKthBiggestGpu(stream, batch, batchBufLen, oLen, multiTop2, threshold);

  batchedFilterNCompressGpu(stream, batch, iLen, batchBufLen, oLen,
                            (Dtype*)ikey, (Dtype*)threshold, (Dtype*)keyBuffer,
                            (unsigned*)idxBuffer, (unsigned*)numFiltered);

  batchedFilteredTop128Gpu(
      stream, batch, iLen, oLen, batchBufLen, (Dtype*)ikey, (Dtype*)keyBuffer,
      (unsigned*)idxBuffer, (Dtype*)multiTop2, (Dtype*)threshold,
      (unsigned*)numFiltered, (Dtype*)okey, (unsigned*)oval);
}

#endif  // DYN_SORT_TOP_K_SORT_H_

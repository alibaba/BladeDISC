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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_RAL_CONTEXT_CONTEXT_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_RAL_CONTEXT_CONTEXT_UTIL_H_

#include <algorithm>

#include "mlir/ral/ral_helper.h"
#include "mlir/ral/ral_logging.h"

namespace tao {
namespace ral {

template <typename T, int N>
inline void print_memref(tao::ral::MemRefType<T, N> memref,
                         const std::string& msg) {
  auto* pmemref = &memref;
  TAO_VLOG(0) << msg << " memref:"
              << "\tbasePtr: " << pmemref->basePtr
              << "\tdata: " << pmemref->data << "\toffset: " << pmemref->offset;
  for (int i = 0; i < N; ++i) {
    TAO_VLOG(0) << "\tsizes" << i << ": " << pmemref->sizes[i];
  }
  for (int i = 0; i < N; ++i) {
    TAO_VLOG(0) << "\tstrides" << i << ": " << pmemref->strides[i];
  }
}

template <typename T>
inline void print_memref_0d(tao::ral::MemRefType<T, 0> memref,
                            const std::string& msg) {
  auto* pmemref = &memref;
  TAO_VLOG(0) << msg << " memref:"
              << "\tbasePtr: " << pmemref->basePtr
              << "\tdata: " << pmemref->data << "\toffset: " << pmemref->offset;
}

template <typename T, int N, typename ShapeTy>
tao::ral::MemRefType<T, N> assignMemRef(void* ptr, const ShapeTy& shape) {
  tao::ral::MemRefType<T, N> memref;
  memref.basePtr = reinterpret_cast<T*>(ptr);
  memref.data = reinterpret_cast<T*>(ptr);
  memref.offset = 0;
  for (int i = 0; i < N; ++i) {
    memref.sizes[i] = shape[i];
  }

  memref.strides[N - 1] = 1;
  for (int i = N - 1; i > 0; --i) {
    memref.strides[i - 1] = memref.strides[i] * memref.sizes[i];
  }

  if (TAO_VLOG_IS_ON(1)) {
    print_memref(memref, "assigned");
  }

  return memref;
}

template <typename T>
tao::ral::MemRefType<T, 0> assignMemRef_0d(void* ptr) {
  tao::ral::MemRefType<T, 0> memref;
  memref.basePtr = reinterpret_cast<T*>(ptr);
  memref.data = reinterpret_cast<T*>(ptr);
  memref.offset = 0;

  if (TAO_VLOG_IS_ON(1)) {
    print_memref_0d(memref, "assigned");
  }

  return memref;
}

template <typename T, int N>
bool isEmptyMemref(tao::ral::MemRefType<T, N> memref) {
  return std::any_of(memref.sizes, memref.sizes + N,
                     [](int64_t dim) { return !dim; });
}

template <typename T>
bool isEmptyMemref(tao::ral::MemRefType<T, 0> memref) {
  return false;
}

template <typename T, int N>
int64_t Size(tao::ral::MemRefType<T, N> memref) {
  int64_t size = 1;
  for (int i = 0; i < N; ++i) {
    size *= memref.sizes[i];
  }
  return size;
}

template <typename T>
int64_t Size(tao::ral::MemRefType<T, 0> memref) {
  return 1;
}

}  // namespace ral
}  // namespace tao

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_RAL_CONTEXT_CONTEXT_UTIL_H_

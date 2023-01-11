//===- ral_base.h ----------------------===//
//
// Copyright 2020 The PAI Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef RAL_RAL_BASE_H_
#define RAL_RAL_BASE_H_

#include <cstdint>
#include <functional>
#include <vector>

namespace tao {
namespace ral {

// integer status type, used for error checking
// value zero is always ok, otherwise is failed
using status_t = int32_t;

// memory buffer abstraction
using buffer_t = void*;

// memory buffer abstraction
using const_buffer_t = const void*;

// opaque resouce abstraction
using opaque_t = void*;

// memory buffer shape abstraction
using buffer_shape_t = std::vector<int64_t>;

// ral api function prototype
using api_func_t = std::function<void(void**)>;

using alloc_t = std::function<buffer_t(size_t)>;
using dealloc_t = std::function<void(buffer_t)>;

struct Allocator {
 public:
  virtual ~Allocator() = default;
  virtual void releaseAllFreeBuffers(){};
  virtual buffer_t alloc(size_t bytes) = 0;
  virtual void dealloc(buffer_t buffer) = 0;
};

}  // namespace ral
}  // namespace tao

#endif  // RAL_RAL_BASE_H_
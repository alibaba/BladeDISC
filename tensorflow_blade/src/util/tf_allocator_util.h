// Copyright 2022 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef __TF_ALLOCATOR_UTIL_H__
#define __TF_ALLOCATOR_UTIL_H__

#include "tensorflow/core/framework/allocator.h"

namespace tf_blade {
namespace util {

using tensorflow::Allocator;

std::pair<int, Allocator*> GetDeviceAndAllocator(int device_id = -1);

// std::align is not supported, so this function mimic its behavior.
void* Align(uint64_t alignment, uint64_t size, void*& ptr, uint64_t& space);

}  // namespace util
}  // namespace tf_blade
#endif  // __TF_ALLOCATOR_UTIL_H__

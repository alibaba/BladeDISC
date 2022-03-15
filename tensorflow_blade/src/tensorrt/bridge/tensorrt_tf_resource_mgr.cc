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

#include "tensorrt_tf_resource_mgr.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorrt_tf_allocator.h"

using tensorflow::AllocatorAttributes;

namespace tf_blade {
namespace trt {

TRTCacheResource::TRTCacheResource(OpKernelContext* ctx) {
  auto device = ctx->device();
  AllocatorAttributes alloc_attr;
  auto alloc = device->GetAllocator(alloc_attr);
  if (!alloc) {
    LOG(ERROR) << "Can't find device allocator for gpu device "
               << device->name();
    allocator_ = nullptr;
  } else {
    allocator_.reset(new TRTDeviceAllocator(alloc));
  }
}

TRTCacheResource::~TRTCacheResource() {
  VLOG(1) << "Destroying TRTCacheResource...";
}

std::string TRTCacheResource::DebugString() const {
  std::stringstream oss;
  using std::dec;
  using std::hex;
  oss << "TRTCacheResource: ";
  oss << "TRTBaseAllocator = " << hex << allocator_.get() << dec;
  return oss.str();
}

}  // namespace trt
}  // namespace tf_blade

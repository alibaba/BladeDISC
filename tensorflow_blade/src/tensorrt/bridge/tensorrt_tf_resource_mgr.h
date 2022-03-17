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

#ifndef __TENSORRT_TF_RESOURCE_MGR_H__
#define __TENSORRT_TF_RESOURCE_MGR_H__

#include <unordered_map>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorrt_tf_allocator.h"

namespace tf_blade {
namespace trt {

using tensorflow::OpKernelContext;
using tensorflow::ResourceBase;

class TRTCacheResource : public ResourceBase {
 public:
  TRTCacheResource(OpKernelContext* ctx);

  ~TRTCacheResource() override;

  std::string DebugString() const override;

  // Keep device allocator for TRT.
  std::unique_ptr<TRTBaseAllocator> allocator_;
};

}  // namespace trt
}  // namespace tf_blade
#endif  //__TENSORRT_TF_RESOURCE_MGR_H__

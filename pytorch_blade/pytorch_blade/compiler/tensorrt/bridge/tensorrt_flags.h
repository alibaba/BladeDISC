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

#ifndef __COMPILER_TENSORRT_FLAGS_H__
#define __COMPILER_TENSORRT_FLAGS_H__

#include "NvInfer.h"
#include "pytorch_blade/common_utils/macros.h"

namespace torch {
namespace blade {

// The TensorRT Engine flags. Note this is per thread
// level setting/getting, for multithreading use cases,
// it needs to be configured separately in each thread.
TorchBladeDeclNewFlag(nvinfer1::BuilderFlags, BuilderFlags);

// The BuilderFlags is thread local.
class BuilderFlagsGuard {
 public:
  BuilderFlagsGuard(nvinfer1::BuilderFlags);
  ~BuilderFlagsGuard();

 private:
  nvinfer1::BuilderFlags prev_flags_;
};

} // namespace blade
} // namespace torch
#endif //__COMPILER_TENSORRT_FLAGS_H__

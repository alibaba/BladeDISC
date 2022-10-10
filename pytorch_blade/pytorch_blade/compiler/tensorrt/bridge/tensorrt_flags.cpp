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

#include "pytorch_blade/compiler/tensorrt/bridge/tensorrt_flags.h"

namespace torch {
namespace blade {

TorchBladeDefNewFlag(nvinfer1::BuilderFlags, BuilderFlags);

// The BuilderFlags is thread local.
BuilderFlagsGuard::BuilderFlagsGuard(nvinfer1::BuilderFlags flags) {
  auto new_flags = GetBuilderFlags();
  new_flags = new_flags | flags;
  prev_flags_ = SetBuilderFlags(new_flags);
}

BuilderFlagsGuard::~BuilderFlagsGuard() {
  SetBuilderFlags(prev_flags_);
}

} // namespace blade
} // namespace torch

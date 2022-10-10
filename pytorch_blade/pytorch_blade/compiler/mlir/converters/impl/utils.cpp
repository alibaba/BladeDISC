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

#include "pytorch_blade/compiler/mlir/converters/impl/utils.h"

#include "pytorch_blade/common_utils/logging.h"
#include "pytorch_blade/compiler/mlir/converters/impl/prim_constant.h"

#include <torch/script.h>

namespace torch {
namespace blade {
bool CheckConstAttribute(
    const torch::jit::Value* attr_val,
    const std::string& op_name,
    const std::string& param_name) {
  if (!IsPrimConstant(attr_val)) {
    TORCH_CHECK(attr_val != nullptr);
    LOG(WARNING) << "Could not convert " << op_name
                 << " with non-compilation time parameter: " << param_name
                 << " %" << attr_val->debugName();
    return false;
  }
  return true;
}
} // namespace blade
} // namespace torch

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

#include "onnx_funcs.h"
#include "pytorch_blade/common_utils/logging.h"

namespace torch {
namespace blade {
using namespace torch::jit;

// For tensorrt7.0, constant operator supports only data types: INT32, FLOAT
void CastDownAllConstantDoubleToFloat(Block* block) {
  auto it = block->nodes().begin();
  for (; it != block->nodes().end(); ++it) {
    auto node = *it;
    for (auto block : node->blocks()) {
      CastDownAllConstantDoubleToFloat(block);
    }

    if (node->kind() == ::c10::onnx::Constant) {
      auto val = node->t(attr::value);
      at::ScalarType dtype = val.scalar_type();
      if (dtype == at::ScalarType::Double) {
        val = val.to(at::ScalarType::Float);
        node->removeAttribute(attr::value);
        node->t_(attr::value, val);
      }
    }
  }
}

void CastDownAllConstantDoubleToFloat(std::shared_ptr<Graph> graph) {
  CastDownAllConstantDoubleToFloat(graph->block());
}
} // namespace blade
} // namespace torch

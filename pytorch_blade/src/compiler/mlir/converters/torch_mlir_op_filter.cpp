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

#include "compiler/mlir/converters/torch_mlir_op_filter.h"
#include <torch/script.h>
#include <string>
#include <unordered_set>
#include "compiler/mlir/converters/mhlo_conversion_context.h"

namespace torch {
namespace blade {
std::unordered_set<std::string> GetTorchMlirWhiteList();

bool IsTorchMlirSupported(const torch::jit::Node& node) {
  auto schema = node.maybeSchema();
  if (schema) {
    std::cout << "schema: " << schema->operator_name().name << std::endl;
    // auto name = schema->operator_name().name;
    return GetTorchMlirWhiteList().find(schema->operator_name().name) !=
        GetTorchMlirWhiteList().end();
  } else if (node.kind().is_prim()) {
    auto name = c10::OperatorName(node.kind().toQualString(), "").name;
    std::cout << "prim: " << name << std::endl;
    return GetTorchMlirWhiteList().find(name) != GetTorchMlirWhiteList().end();
  }
  std::cout << "not supported" << std::endl;
  return false;
}

std::unordered_set<std::string> GetTorchMlirWhiteList() {
  return std::unordered_set<std::string>{
      "aten::relu",
      "aten::relu6",
      "aten::leaky_relu",
      "aten::sigmoid",
      "aten::glu",
      "aten::contiguous",
      "aten::erf",
      "aten::exp",
      "aten::neg",
      "aten::rsqrt",
      "aten::neg",
      "aten::to.dtype",
      "aten::type_as",
      "aten::bitwise_not",
      "aten::to",
  };
}

} //  namespace blade
} //  namespace torch
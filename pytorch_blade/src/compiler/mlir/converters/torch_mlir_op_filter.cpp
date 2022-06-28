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
    return GetTorchMlirWhiteList().find(schema->operator_name().name) !=
        GetTorchMlirWhiteList().end();
  } else if (node.kind().is_prim()) {
    auto name = c10::OperatorName(node.kind().toQualString(), "").name;
    return GetTorchMlirWhiteList().find(name) != GetTorchMlirWhiteList().end();
  }
  return false;
}

// clang-format off
std::unordered_set<std::string> GetTorchMlirWhiteList() {
  return std::unordered_set<std::string>{
      "aten::__and__",
      "aten::add",
      "aten::addmm",
      "aten::bind",
      "aten::bitwise_not",
      "aten::bmm",
      "aten::chunk",
      "aten::contiguous",
      "aten::div",
      "aten::eq",
      "aten::erf",
      "aten::exp",
      "aten::expand",
      "aten::expand_as",
      "aten::flatten",
      "aten::floor_divide",
      "aten::gelu",
      "aten::glu",
      "aten::hardtanh",
      "aten::leaky_relu",
      "aten::linear",
      "aten::lt",
      "aten::matmul",
      "aten::mean",
      "aten::mm",
      "aten::mul",
      "aten::ne",
      "aten::neg",
      "aten::neg",
      "aten::permute",
      "aten::relu",
      "aten::relu6",
      "aten::repeat",
      "aten::reshape",
      "aten::rsqrt",
      "aten::select",
      "aten::sigmoid",
      "aten::silu",
      "aten::size",
      "aten::slice",
      "aten::sub",
      "aten::sum",
      "aten::t",
      "aten::to",
      "aten::to.dtype",
      "aten::transpose",
      "aten::type_as",
      "aten::view",
      "aten::view_as",
      "prim::Constant",
      "prim::ListConstruct",
      "prim::ListUnpack"};
}
// clang-format off

} //  namespace blade
} //  namespace torch

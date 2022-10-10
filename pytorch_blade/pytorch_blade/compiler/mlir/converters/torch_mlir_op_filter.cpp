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

#include "pytorch_blade/compiler/mlir/converters/torch_mlir_op_filter.h"

#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include "pytorch_blade/common_utils/utils.h"
#include "pytorch_blade/compiler/mlir/converters/mhlo_conversion_context.h"

#include <torch/script.h>

namespace torch {
namespace blade {
const std::unordered_set<std::string>& GetTorchMlirWhiteList();

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
const std::unordered_set<std::string> &GetTorchMlirWhiteList() {
  static std::unordered_set<std::string> white_list{
      "aten::_autocast_to_reduced_precision",
      "aten::__and__",
      "aten::add",
      "aten::addmm",
      "aten::arange",
      "aten::unbind",
      "aten::batch_norm",
      "aten::bitwise_not",
      "aten::bmm",
      "aten::cat",
      "aten::chunk",
      "aten::contiguous",
      "aten::_convolution",
      "aten::convolution",
      "aten::conv1d",
      "aten::conv2d",
      "aten::div",
      "aten::embedding",
      "aten::eq",
      "aten::gt",
      "aten::ge",
      "aten::lt",
      "aten::le",
      "aten::erf",
      "aten::exp",
      "aten::expand",
      "aten::expand_as",
      "aten::flatten",
      "aten::flip",
      "aten::floor_divide",
      "aten::gelu",
      "aten::gelu_backward",
      "aten::glu",
      "aten::hardsigmoid",
      "aten::hardswish",
      "aten::hardtanh",
      "aten::index_select",
      "aten::item",
      "aten::Int",
      "aten::layer_norm",
      "aten::leaky_relu",
      "aten::linear",
      "aten::log_softmax",
      "aten::lt",
      "aten::matmul",
      "aten::masked_fill",
      "aten::mean",
      "aten::mm",
      "aten::mul",
      "aten::native_layer_norm",
      "aten::ne",
      "aten::neg",
      "aten::neg",
      "aten::native_dropout",
      "aten::permute",
      "aten::pow",
      "aten::relu",
      "aten::relu6",
      "aten::repeat",
      "aten::reshape",
      "aten::roll",
      "aten::rsqrt",
      "aten::select",
      "aten::sigmoid",
      "aten::silu",
      "aten::size",
      "aten::slice",
      "aten::softmax",
      "aten::std",
      "aten::squeeze",
      "aten::sub",
      "aten::sum",
      "aten::ScalarImplicit",
      "aten::t",
      "aten::tensor",
      "aten::to",
      "aten::to.dtype",
      "aten::transpose",
      "aten::type_as",
      "aten::unsqueeze",
      "aten::view",
      "aten::view_as",
      "prim::Constant",
      "prim::ListConstruct",
      "prim::ListUnpack",
      // Torch Blade custom ops follows:
      "torch_blade::fake_quant"
    };


  static std::once_flag white;
  std::call_once(white, []() {
      auto custom_ops = env::ReadStringFromEnvVar("TORCH_MHLO_OP_WHITE_LIST", "");
      std::ostringstream ostr;
      ostr << "User defined white list: [";
      std::istringstream f(custom_ops);
      std::string s;
      for (auto s : StrSplit(custom_ops, ';')) {
        white_list.insert(std::string(s));
        ostr << s << ", ";
      }
      ostr << "]";
      LOG(INFO) << ostr.str();
  });
  return white_list;
}
// clang-format off

} //  namespace blade
} //  namespace torch

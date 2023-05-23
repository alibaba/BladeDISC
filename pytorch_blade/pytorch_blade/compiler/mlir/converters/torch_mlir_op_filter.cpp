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

#include <torch/script.h>

namespace torch {
namespace blade {
const std::unordered_set<std::string>& GetTorchMlirWhiteList();

bool IsTorchMlirSupported(const torch::jit::Node& node) {
  auto schema = node.maybeSchema();
  bool ret = false;
  if (schema) {
    ret = GetTorchMlirWhiteList().find(schema->operator_name().name) !=
        GetTorchMlirWhiteList().end();
  } else if (node.kind().is_prim()) {
    auto name = c10::OperatorName(node.kind().toQualString(), "").name;
    ret = GetTorchMlirWhiteList().find(name) != GetTorchMlirWhiteList().end();
  }
  return ret;
}

// clang-format off
const std::unordered_set<std::string> &GetTorchMlirWhiteList() {
  static std::unordered_set<std::string> white_list{
#if defined(PLATFORM_ALIBABA) and defined(ENABLE_BLADE_GEMM)
      "aten::upsample_nearest2d.vec",
#endif
      "aten::_autocast_to_reduced_precision",
      "aten::_autocast_to_full_precision",
      "aten::__and__",
      "aten::__getitem__",
      "aten::_softmax",
      "aten::abs",
      "aten::add",
      "aten::addmm",
      "aten::arange",
      "aten::unbind",
      "aten::baddbmm",
      "aten::baddmm",
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
      "aten::cos",
      "aten::div",
      "aten::detach",
      "aten::einsum",
      "aten::embedding",
      "aten::empty",
      "aten::eq",
      "aten::flip",
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
      "aten::full",
      "aten::gather",
      "aten::gelu",
      "aten::gelu_backward",
      "aten::glu",
      "aten::group_norm",
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
      "aten::max",
      "aten::mean",
      "aten::mm",
      "aten::mul",
      "aten::narrow",
      "aten::native_layer_norm",
      "aten::ne",
      "aten::neg",
      "aten::neg",
      // TODO(disc): need to lower mhlo::RngOp to mhlo_disc::UniformOp
      //"aten::native_dropout",
      "aten::ones",
      "aten::ones_like",
      "aten::pad",
      "aten::permute",
      "aten::pow",
      "aten::relu",
      "aten::relu6",
      "aten::repeat",
      "aten::reshape",
      "aten::roll",
      "aten::rsqrt",
      "aten::rsub",
      "aten::select",
      "aten::selu",
      "aten::selu_",
      "aten::sigmoid",
      "aten::silu",
      "aten::sin",
      "aten::size",
      "aten::slice",
      "aten::softmax",
      "aten::split",
      "aten::std",
      "aten::squeeze",
      "aten::sub",
      "aten::sum",
      "aten::ScalarImplicit",
      "aten::t",
      "aten::tanh",
      "aten::tensor",
      "aten::to",
      "aten::to.dtype",
      "aten::to.device",
      "aten::transpose",
      "aten::type_as",
      "aten::unsqueeze",
      "aten::view",
      "aten::view_as",
      "aten::where",
      "aten::zeros",
      "aten::zeros_like",
      "prim::device",
      "prim::dtype",
      "prim::Constant",
      "prim::ListConstruct",
      "prim::ListUnpack",
      "prim::NumToTensor",
      // Torch Blade custom ops follows:
      "aten::add_inplace", // use aten namespace to work with PyTorch mutation pass
      "aten::sub_inplace", // use aten namespace to work with PyTorch mutation pass
      "aten::mul_inplace", // use aten namespace to work with PyTorch mutation pass
      "aten::div_inplace", // use aten namespace to work with PyTorch mutation pass
      "torch_blade::fake_quant"
    };

  static std::once_flag white, black;

  std::call_once(white, [&]() {
    auto list = StrSplit(env::ReadStringFromEnvVar("TORCH_MHLO_OP_WHITE_LIST", ""), ';');
    for (auto s : list) {
      white_list.insert(s);
    }
  });
  std::call_once(black, [&]() {
    auto list = StrSplit(env::ReadStringFromEnvVar("TORCH_MHLO_OP_BLACK_LIST", ""), ';');
    for (auto s : list) {
      white_list.erase(s);
    }
  });

  std::ostringstream ostr;
  ostr << "User defined black list: [";
  for (auto op : white_list) {
    ostr << op << ", ";
  }
  LOG(INFO) << ostr.str() << "]";

  return white_list;
}
// clang-format off

} //  namespace blade
} //  namespace torch

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

#include <mlir/mhlo/builder/mlir_type_utils.h>

#include "pytorch_blade/common_utils/logging.h"
#include "pytorch_blade/compiler/mlir/converters/impl/prim_constant.h"
#include "pytorch_blade/compiler/mlir/converters/impl/utils.h"
#include "pytorch_blade/compiler/mlir/converters/mhlo_conversion_context.h"
#include "pytorch_blade/compiler/mlir/converters/mhlo_converter_register.h"

#include <torch/script.h>
namespace torch {
namespace blade {

bool ConvertPrimListConstruct(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  auto list_out = node.output();
  auto list_type = list_out->type()->cast<c10::ListType>();
  auto supported_type = false;
  if (list_type) {
    auto elem_type = list_type->getElementType();
    supported_type = elem_type == c10::TensorType::get() ||
        elem_type == c10::IntType::get() ||
        elem_type == c10::FloatType::get() || elem_type == c10::BoolType::get();
  }
  if (!supported_type) {
    return false;
  }

  if (ctx.IsSupportTesting()) {
    return true;
  }

  auto loc = GetNodeLocation(ctx, node);
  mlir::mhlo::SmallVec4<mlir::Value> list_vals;
  list_vals.reserve(node.inputs().size());
  for (auto inp : node.inputs()) {
    auto val = ctx.GetMlirValue(inp);
    list_vals.push_back(val);
  }
  ctx.list_map[node.output()] = list_vals;
  return true;
}

bool ConvertAtenGetItem(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  auto loc = GetNodeLocation(ctx, node);
  auto jit_vals = node.input(0);
  if (ctx.list_map.find(jit_vals) == ctx.list_map.end()) {
    return false;
  }
  auto jit_idx = node.input(1);
  if (!CheckConstAttribute(jit_idx, "aten::__getitem__", "idx")) {
    return false;
  }
  auto list_vals = ctx.GetMlirValueList(jit_vals);
  auto index = CastJitConstToInt64(*jit_idx);
  if (index >= list_vals.size()) {
    return false;
  }
  ctx.value_map[node.output(0)] = list_vals[index];
  return true;
}

bool ConvertPrimListUnpack(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  if (ctx.IsSupportTesting()) {
    return true;
  }
  auto loc = GetNodeLocation(ctx, node);
  auto jit_vals = node.input(0);
  if (ctx.list_map.find(jit_vals) == ctx.list_map.end()) {
    return false;
  }

  auto& inp_vals = ctx.list_map[jit_vals];
  if (inp_vals.size() != node.outputs().size()) {
    return false;
  }
  for (size_t k = 0; k < inp_vals.size(); ++k) {
    ctx.value_map[node.output(k)] = inp_vals[k];
  }
  return true;
}

namespace {
auto mhlo_conversion =
    MhloConversionPatternRegister()
        .pattern(
            GetPrimOperatorName(prim::ListConstruct),
            ConvertPrimListConstruct)
        .pattern(GetPrimOperatorName(prim::ListUnpack), ConvertPrimListUnpack)
        .pattern(
            "aten::__getitem__.t(t[](a) list, int idx) -> (t(*))",
            ConvertAtenGetItem);
}
} // namespace blade
} // namespace torch

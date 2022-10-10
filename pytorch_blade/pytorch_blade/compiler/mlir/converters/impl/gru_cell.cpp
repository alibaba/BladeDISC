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

#include <mlir/mhlo/builder/gru_cell.h>
#include <mlir/mhlo/builder/matmul.h>

#include "pytorch_blade/compiler/mlir/converters/impl/prim_constant.h"
#include "pytorch_blade/compiler/mlir/converters/impl/utils.h"
#include "pytorch_blade/compiler/mlir/converters/mhlo_converter_register.h"

#include <torch/script.h>

namespace torch {
namespace blade {
bool ConvertAtenGRUCell(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  const char* op_name = "aten::gru_cell";
  auto jit_input_weight = node.input(2);
  if (!CheckConstAttribute(jit_input_weight, op_name, "w_ih")) {
    return false;
  }
  auto jit_hidden_weight = node.input(3);
  if (!CheckConstAttribute(jit_hidden_weight, op_name, "w_hh")) {
    return false;
  }

  auto jit_input_bias = node.input(4);
  auto jit_input_bias_ival = torch::jit::toIValue(jit_input_bias);
  if (jit_input_bias_ival && jit_input_bias_ival->isNone()) {
    return false;
  }
  auto jit_hidden_bias = node.input(5);
  auto jit_hidden_bias_ival = torch::jit::toIValue(jit_hidden_bias);
  if (jit_hidden_bias_ival && jit_hidden_bias_ival->isNone()) {
    return false;
  }

  auto torch_input_weight = torch::jit::toIValue(jit_input_weight);
  auto torch_hidden_weight = torch::jit::toIValue(jit_hidden_weight);
  if (!(torch_input_weight && torch_hidden_weight)) {
    return false;
  }
  if (!(torch_input_weight->isTensor() && torch_hidden_weight->isTensor())) {
    return false;
  }
  at::Tensor input_weight = torch_input_weight->toTensor();
  at::Tensor hidden_weight = torch_hidden_weight->toTensor();
  const auto& loc = GetNodeLocation(ctx, node);
  auto& builder = *ctx.builder;
  auto ml_w_ih_t =
      BuildMlirConstFromTorchTensor(builder, loc, input_weight.t());
  auto ml_w_hh_t =
      BuildMlirConstFromTorchTensor(builder, loc, hidden_weight.t());

  auto ml_input = ctx.GetMlirValue(node.input(0));
  auto ml_h_x = ctx.GetMlirValue(node.input(1));
  auto ml_input_gates =
      mlir::mhlo::BuildDotProduct_mm(builder, loc, ml_input, ml_w_ih_t);
  auto ml_hidden_gates =
      mlir::mhlo::BuildDotProduct_mm(builder, loc, ml_h_x, ml_w_hh_t);

  auto ml_input_bias = ctx.GetMlirValue(jit_input_bias);
  auto ml_hidden_bias = ctx.GetMlirValue(jit_hidden_bias);
  ctx.value_map[node.output(0)] = mlir::mhlo::BuildGRUCell(
      builder,
      loc,
      ml_input_gates,
      ml_hidden_gates,
      ml_h_x,
      ml_input_bias,
      ml_hidden_bias);
  return true;
}

bool ConvertAtenFusedGRUCell(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  auto jit_input_bias = node.input(3);
  auto jit_input_bias_ival = torch::jit::toIValue(jit_input_bias);
  if (jit_input_bias_ival && jit_input_bias_ival->isNone()) {
    return false;
  }
  auto jit_hidden_bias = node.input(4);
  auto jit_hidden_bias_ival = torch::jit::toIValue(jit_hidden_bias);
  if (jit_hidden_bias_ival && jit_hidden_bias_ival->isNone()) {
    return false;
  }

  auto ml_input_gates = ctx.GetMlirValue(node.input(0));
  auto ml_hidden_gates = ctx.GetMlirValue(node.input(1));
  auto ml_h_x = ctx.GetMlirValue(node.input(2));
  auto ml_input_bias = ctx.GetMlirValue(jit_input_bias);
  auto ml_hidden_bias = ctx.GetMlirValue(jit_hidden_bias);
  auto& builder = *ctx.builder;
  const auto& loc = GetNodeLocation(ctx, node);
  ctx.value_map[node.output(0)] = mlir::mhlo::BuildGRUCell(
      builder,
      loc,
      ml_input_gates,
      ml_hidden_gates,
      ml_h_x,
      ml_input_bias,
      ml_hidden_bias);
  return true;
}

namespace {
auto mhlo_conversion =
    MhloConversionPatternRegister()
        .pattern(
            R"SIG(aten::_thnn_fused_gru_cell(
            Tensor input_gates, Tensor hidden_gates, Tensor hx,
            Tensor? input_bias=None, Tensor? hidden_bias=None) -> (Tensor, Tensor))SIG",
            ConvertAtenFusedGRUCell)
        .pattern(
            R"SIG(aten::gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> (Tensor))SIG",
            ConvertAtenGRUCell);
}
} // namespace blade
} // namespace torch

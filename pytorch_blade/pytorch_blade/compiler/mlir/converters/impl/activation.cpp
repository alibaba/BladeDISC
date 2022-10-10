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

#include <mlir-hlo/Dialect/mhlo/IR/hlo_ops.h> // from tf repo
#include <mlir/mhlo/builder/activation.h>
#include <mlir/mhlo/builder/element_wise_binary.h>
#include <mlir/mhlo/builder/mlir_type_utils.h>
#include <mlir/mhlo/builder/slice.h>
#include <mlir/mhlo/builder/standard.h>

#include "pytorch_blade/compiler/mlir/converters/impl/prim_constant.h"
#include "pytorch_blade/compiler/mlir/converters/impl/utils.h"
#include "pytorch_blade/compiler/mlir/converters/mhlo_converter_register.h"

#include <torch/script.h>

namespace torch {
namespace blade {

bool ConvertAtenHardtanh(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  auto loc = GetNodeLocation(ctx, node);
  auto inp_val = ctx.GetMlirValue(node.input(0));
  auto min_val = ctx.GetMlirValue(node.input(1));
  auto max_val = ctx.GetMlirValue(node.input(2));

  auto builder = *ctx.builder;
  auto elem_type = mlir::mhlo::GetMlirTensorElemType(inp_val);
  auto lb =
      mlir::mhlo::BuildStdScalarToHloTensor(builder, loc, min_val, elem_type);
  auto ub =
      mlir::mhlo::BuildStdScalarToHloTensor(builder, loc, max_val, elem_type);
  auto result = builder.create<mlir::mhlo::ClampOp>(
      loc, inp_val.getType(), lb, inp_val, ub);

  ctx.value_map[node.output(0)] = result.getResult();
  return true;
}

bool ConvertAtenRelu(MhloConversionContext& ctx, const torch::jit::Node& node) {
  const auto& loc = GetNodeLocation(ctx, node);
  const auto& ml_input = ctx.GetMlirValue(node.input(0));
  auto builder = *ctx.builder;
  auto elem_type = mlir::mhlo::GetMlirTensorElemType(ml_input);
  auto zero = mlir::mhlo::BuildHloConstZeroForType(builder, loc, elem_type);
  const auto& relu = mlir::mhlo::BuildMlirBinaryOp<mlir::chlo::BroadcastMaxOp>(
      builder, loc, ml_input, zero, elem_type);
  ctx.value_map[node.output(0)] = relu;
  return true;
}

bool ConvertAtenLeakyRelu(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  const auto& loc = GetNodeLocation(ctx, node);
  const auto& ml_input = ctx.GetMlirValue(node.input(0));
  const auto& negative_slope = node.input(1);
  if (!IsPrimConstant(negative_slope)) {
    return false;
  }
  auto builder = *ctx.builder;
  auto elem_type = mlir::mhlo::GetMlirTensorElemType(ml_input);
  auto zero = mlir::mhlo::BuildHloConstZeroForType(builder, loc, elem_type);
  const auto& positive_part =
      mlir::mhlo::BuildMlirBinaryOp<mlir::chlo::BroadcastMaxOp>(
          builder, loc, ml_input, zero, elem_type);
  mlir::Value negative_part =
      mlir::mhlo::BuildMlirBinaryOp<mlir::chlo::BroadcastMinOp>(
          builder, loc, ml_input, zero, elem_type);
  mlir::Value negative_slope_value = mlir::mhlo::BuildHloConstForFloatType(
      builder, loc, elem_type, CastJitConstToDouble(*negative_slope));
  negative_part = mlir::mhlo::BuildMlirBinaryOp<mlir::chlo::BroadcastMulOp>(
      builder, loc, negative_part, negative_slope_value, elem_type);
  mlir::Value out = mlir::mhlo::BuildMlirBinaryOp<mlir::chlo::BroadcastAddOp>(
      builder, loc, positive_part, negative_part, elem_type, true);

  ctx.value_map[node.output(0)] = out;
  return true;
}

bool ConvertAtenGlu(MhloConversionContext& ctx, const torch::jit::Node& node) {
  // ref:
  // https://pytorch.org/docs/stable/nn.functional.html?highlight=glu#torch.nn.functional.glu
  const auto& loc = GetNodeLocation(ctx, node);
  const auto& ml_input = ctx.GetMlirValue(node.input(0));
  const auto& jit_dim = node.input(1);
  // TODO: dynamic dim could be support, implement it if need.
  // Currently, we only support const dim, so that
  // the negtive dim index is easy to be normalized at converter time.
  if (!CheckConstAttribute(jit_dim, "aten::glu", "dim")) {
    return false;
  }
  auto dim_index = CastJitConstToInt64(*jit_dim);
  auto builder = *ctx.builder;
  mlir::Value lhs, rhs;
  std::tie(lhs, rhs) =
      mlir::mhlo::BuildHalfSplit(builder, loc, ml_input, dim_index);
  rhs = mlir::mhlo::BuildSigmoid(builder, loc, rhs);
  auto elem_type = mlir::mhlo::GetMlirTensorElemType(ml_input);
  auto glu = mlir::mhlo::BuildMlirBinaryOp<mlir::chlo::BroadcastMulOp>(
      builder, loc, lhs, rhs, elem_type);
  ctx.value_map[node.output()] = glu;
  return true;
}

bool ConvertAtenSigmoid(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  auto loc = GetNodeLocation(ctx, node);
  auto torch_inp = node.input(0);
  auto torch_out = node.output(0);
  auto mlir_val = ctx.GetMlirValue(torch_inp);
  ctx.value_map[torch_out] =
      mlir::mhlo::BuildSigmoid(*ctx.builder, loc, mlir_val);
  return true;
}

bool ConvertAtenGelu(MhloConversionContext& ctx, const torch::jit::Node& node) {
  // ref:
  // https://pytorch.org/docs/stable/nn.functional.html?highlight=gelu#torch.nn.functional.gelu
  // gelu(x) = x * 1/2 * [1 + erf(x/(sqrt(2)))]
  const auto& loc = GetNodeLocation(ctx, node);
  const auto& ml_input = ctx.GetMlirValue(node.input(0));
  auto builder = *ctx.builder;
  auto elem_type = mlir::mhlo::GetMlirTensorElemType(ml_input);
  auto one = mlir::mhlo::BuildHloConstForFloatType(builder, loc, elem_type, 1);
  auto two = mlir::mhlo::BuildHloConstForFloatType(builder, loc, elem_type, 2);
  auto half =
      mlir::mhlo::BuildHloConstForFloatType(builder, loc, elem_type, 0.5);
  auto rsqrt_two = builder.create<mlir::mhlo::RsqrtOp>(loc, two);
  auto erf_element = mlir::mhlo::BuildMlirBinaryOp<mlir::chlo::BroadcastMulOp>(
      builder, loc, ml_input, rsqrt_two, elem_type);
  auto erf = builder.create<mlir::chlo::ErfOp>(loc, erf_element);
  auto erf_add = mlir::mhlo::BuildMlirBinaryOp<mlir::chlo::BroadcastAddOp>(
      builder, loc, erf, one, elem_type);
  auto half_mul = mlir::mhlo::BuildMlirBinaryOp<mlir::chlo::BroadcastMulOp>(
      builder, loc, erf_add, half, elem_type);
  auto result = builder.create<mlir::mhlo::MulOp>(loc, ml_input, half_mul);
  ctx.value_map[node.output(0)] = result;
  return true;
}

bool ConvertAtenSilu(MhloConversionContext& ctx, const torch::jit::Node& node) {
  auto loc = GetNodeLocation(ctx, node);
  auto torch_inp = node.input(0);
  auto torch_out = node.output(0);
  auto mlir_val = ctx.GetMlirValue(torch_inp);
  auto builder = *ctx.builder;
  auto sigmoid = mlir::mhlo::BuildSigmoid(builder, loc, mlir_val);
  auto silu = builder.create<mlir::mhlo::MulOp>(loc, mlir_val, sigmoid);
  ctx.value_map[torch_out] = silu;
  return true;
}

namespace {
auto mhlo_conversion =
    MhloConversionPatternRegister()
        .pattern("aten::relu(Tensor self) -> Tensor", ConvertAtenRelu)
        .pattern(
            "aten::leaky_relu(Tensor self, Scalar negative_slope=0.01) -> "
            "(Tensor)",
            ConvertAtenLeakyRelu)
        .pattern(
            "aten::hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor",
            ConvertAtenHardtanh)
        .pattern("aten::sigmoid(Tensor self) -> Tensor", ConvertAtenSigmoid)
        .pattern("aten::glu(Tensor self, int dim=-1) -> Tensor", ConvertAtenGlu)
        .pattern("aten::gelu(Tensor self) -> Tensor", ConvertAtenGelu)
        .pattern("aten::silu(Tensor self) -> Tensor", ConvertAtenSilu);
} // namespace
} // namespace blade
} // namespace torch

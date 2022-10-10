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

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "pytorch_blade/compiler/mlir/converters/mhlo_converter_register.h"
#include "pytorch_blade/compiler/mlir/converters/mlir_type_utils.h"
#include "stablehlo/dialect/ChloOps.h"

#include <torch/script.h>

namespace torch {
namespace blade {

template <class MLIR_UNARY_OP>
bool ConvertAtenUnaryOp(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  auto loc = GetNodeLocation(ctx, node);
  auto torch_inp = node.input(0);
  auto torch_out = node.output(0);
  auto mlir_val = ctx.GetMlirValue(torch_inp);

  auto result = ctx.builder->create<MLIR_UNARY_OP>(loc, mlir_val);

  ctx.value_map[torch_out] = result.getResult();
  return true;
}

bool ConvertAtenIdentity(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  const auto& ml_input = ctx.GetMlirValue(node.input(0));
  ctx.value_map[node.output(0)] = ml_input;
  return true;
}

bool ConvertAtenToDtype(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  auto loc = GetNodeLocation(ctx, node);
  auto torch_inp = node.input(0);
  auto torch_out = node.output(0);
  auto dtype_value = torch::jit::toIValue(node.input(1));
  if (!dtype_value || dtype_value->isNone()) {
    return false;
  }
  // TODO: if the target type is the same with input type, return identity.
  // ScalarType in jit::Graph is type of int
  at::ScalarType dtype = static_cast<at::ScalarType>(dtype_value->toInt());
  auto input_val = ctx.GetMlirValue(torch_inp);
  auto elem_type = BuildMlirElemType(*ctx.builder, dtype);
  auto output_val =
      ctx.builder->create<mlir::mhlo::ConvertOp>(loc, input_val, elem_type);
  ctx.value_map[torch_out] = output_val.getResult();
  return true;
}

bool ConvertAtenTypeAs(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  auto loc = GetNodeLocation(ctx, node);
  auto torch_inp = node.input(0);
  auto torch_other = node.input(1);

  auto input_val = ctx.GetMlirValue(torch_inp);
  auto other_val = ctx.GetMlirValue(torch_other);
  auto elem_type =
      other_val.getType().cast<mlir::RankedTensorType>().getElementType();
  auto output_val =
      ctx.builder->create<mlir::mhlo::ConvertOp>(loc, input_val, elem_type);
  ctx.value_map[node.output(0)] = output_val.getResult();
  return true;
}

namespace {
auto mhlo_conversion =
    MhloConversionPatternRegister()
        .pattern(
            "aten::tanh(Tensor self) -> Tensor",
            ConvertAtenUnaryOp<mlir::mhlo::TanhOp>)
        .pattern(
            "aten::neg(Tensor self) -> Tensor",
            ConvertAtenUnaryOp<mlir::mhlo::NegOp>)
        .pattern(
            "aten::exp(Tensor self) -> Tensor",
            ConvertAtenUnaryOp<mlir::mhlo::ExpOp>)
        .pattern(
            "aten::rsqrt(Tensor self) -> Tensor",
            ConvertAtenUnaryOp<mlir::mhlo::RsqrtOp>)
        .pattern(
            "aten::erf(Tensor self) -> Tensor",
            ConvertAtenUnaryOp<mlir::chlo::ErfOp>)
        .pattern(
            "aten::bitwise_not(Tensor self) -> Tensor",
            ConvertAtenUnaryOp<mlir::mhlo::NotOp>)
        .pattern(
            "aten::contiguous(Tensor self, *, MemoryFormat "
            "memory_format=contiguous_format) -> Tensor",
            ConvertAtenIdentity)
        .pattern(
            "aten::to.dtype(Tensor self, int dtype, bool "
            "non_blocking=False, bool copy=False, int? "
            "memory_format=None) -> (Tensor)",
            ConvertAtenToDtype)
        .pattern(
            "aten::to.dtype_layout(Tensor(a) self, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, bool copy=False, int? memory_format=None) -> (Tensor(a))",
            ConvertAtenToDtype)
        .pattern(
            "aten::cuda(Tensor(a) self) -> (Tensor(b|a))",
            ConvertAtenIdentity)
        .pattern(
            "aten::cpu(Tensor(a) self) -> (Tensor(b|a))",
            ConvertAtenIdentity)
        .pattern(
            "aten::clone(Tensor self, *, int? memory_format=None) -> (Tensor)",
            ConvertAtenIdentity)
        .pattern(
            "aten::detach(Tensor(a) self) -> (Tensor(a))",
            ConvertAtenIdentity)
        .pattern(
            "aten::type_as(Tensor self, Tensor other) -> Tensor",
            ConvertAtenTypeAs);
}
} // namespace blade
} // namespace torch

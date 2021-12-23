#include <mlir-hlo/Dialect/mhlo/IR/chlo_ops.h>
#include "compiler/mlir/converters/mhlo_converter_register.h"
#include "compiler/mlir/converters/mlir_type_utils.h"

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
  // ScalarType in jit::Graph is type of int
  torch::ScalarType dtype =
      static_cast<torch::ScalarType>(dtype_value->toInt());
  auto input_val = ctx.GetMlirValue(torch_inp);
  auto elem_type = BuildMlirElemType(*ctx.builder, dtype);
  auto output_val =
      ctx.builder->create<mlir::mhlo::ConvertOp>(loc, input_val, elem_type);
  ctx.value_map[torch_out] = output_val.getResult();
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
            "aten::contiguous(Tensor self, *, MemoryFormat "
            "memory_format=contiguous_format) -> Tensor",
            ConvertAtenIdentity)
        .pattern(
            "aten::to.dtype(Tensor self, int dtype, bool "
            "non_blocking=False, bool copy=False, int? "
            "memory_format=None) -> (Tensor)",
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
            ConvertAtenIdentity);
}
} // namespace blade
} // namespace torch

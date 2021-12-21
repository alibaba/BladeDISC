#include <mlir/mhlo/builder/algebra_statistics.h>
#include <mlir/mhlo/builder/mlir_shape_builder.h>
#include <mlir/mhlo/builder/mlir_utils.h>

#include "common_utils/logging.h"
#include "compiler/mlir/converters/impl/prim_constant.h"
#include "compiler/mlir/converters/impl/utils.h"
#include "compiler/mlir/converters/mhlo_converter_register.h"

#include <torch/script.h>

namespace torch {
namespace addons {
using namespace mlir::mhlo;

bool ConvertAtenLayerNorm(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  // Ref: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
  const auto ml_input = ctx.GetMlirValue(node.input(0));
  const auto& jit_norm_shape = node.input(1);
  const auto& jit_weight = node.input(2);
  const auto& jit_bias = node.input(3);
  const auto& jit_eps_scalar = node.input(4);

  // We use IsPrimConstant here because some prim constant is not supported.
  // So that they have no counter part mlir::Value.
  const char* op_name = node.kind().toDisplayString();
  bool is_const_attr = CheckConstAttribute(jit_weight, op_name, "weight") &&
      CheckConstAttribute(jit_bias, op_name, "bias") &&
      CheckConstAttribute(jit_eps_scalar, op_name, "eps");
  if (!is_const_attr) {
    return false;
  }
  // The mean and standard-deviation are calculated separately over the last
  // certain number dimensions which have to be of the shape specified by
  // normalized_shape.
  //
  // NB: we only use the rank of normalized_shape, because normalized_shape must
  // be exactly the same as the last certain number dimensions of the input
  // tensor, which should have been validated by PyTorch Interpreter.
  mlir_dim_t input_rank = GetRankOfMlirValue(ml_input);
  if (ctx.list_map.find(jit_norm_shape) == ctx.list_map.end()) {
    return false;
  }
  auto ml_norm_shape = ctx.GetMlirValueList(jit_norm_shape);
  if (input_rank < ml_norm_shape.size()) {
    DLOG(INFO) << " Could not convert " << op_name
               << " with normalized_shape has rank greater than the input: "
               << ml_norm_shape.size() << " vs " << input_rank;
    return false;
  }

  auto& builder = *ctx.builder;
  const auto& loc = GetNodeLocation(ctx, node);
  auto norm_input = BuildStandardNorm(
      builder,
      loc,
      ml_input,
      CastJitConstToDouble(*jit_eps_scalar),
      ml_norm_shape.size());
  ::llvm::Optional<mlir::Value> affine_weight =
      ctx.GetOptionalMlirValue(jit_weight);
  ::llvm::Optional<mlir::Value> affine_bias =
      ctx.GetOptionalMlirValue(jit_bias);
  ctx.value_map[node.output(0)] = BuildElemAffine(
      builder, loc, norm_input, affine_weight, affine_bias, ::llvm::None);
  return true;
}

namespace {
auto mhlo_conversion = MhloConversionPatternRegister().pattern(
    R"SIG(aten::layer_norm(Tensor input, int[] normalized_shape,
                           Tensor? weight=None, Tensor? bias=None,
                           float eps=1e-05, bool cudnn_enable=True) -> Tensor)SIG",
    ConvertAtenLayerNorm);
} // namespace
} // namespace addons
} // namespace torch

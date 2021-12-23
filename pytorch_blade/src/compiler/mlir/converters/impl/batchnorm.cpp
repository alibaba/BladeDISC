#include <mlir/mhlo/builder/algebra_statistics.h>
#include <mlir/mhlo/builder/mlir_utils.h>

#include "compiler/mlir/converters/impl/prim_constant.h"
#include "compiler/mlir/converters/impl/utils.h"
#include "compiler/mlir/converters/mhlo_converter_register.h"

// was included last because of llvm headers conflicts
#include <torch/script.h>

namespace torch {
namespace blade {
using namespace mlir::mhlo;

bool ConvertAtenBatchNorm(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  const char* op_name = node.kind().toDisplayString();
  auto jit_training = node.input(5);
  if (!CheckConstAttribute(jit_training, op_name, "training")) {
    return false;
  }
  if (CastJitConstToBool(*jit_training)) {
    // currently training version not support
    return false;
  }
  auto jit_eps = node.input(7);
  if (!CheckConstAttribute(jit_eps, op_name, "eps")) {
    return false;
  }

  auto& builder = *ctx.builder;
  const auto& loc = GetNodeLocation(ctx, node);

  auto ml_input = ctx.GetMlirValue(node.input(0));
  mlir_dim_t input_rank = GetRankOfMlirValue(ml_input);
  SmallVec4<mlir_dim_t> broadcast_dims{1};

  ::llvm::Optional<mlir::Value> running_mean =
      ctx.GetOptionalMlirValue(node.input(3));
  ::llvm::Optional<mlir::Value> running_var =
      ctx.GetOptionalMlirValue(node.input(4));
  auto norm_input = BuildStandardNorm(
      builder,
      loc,
      ml_input,
      running_var,
      running_mean,
      CastJitConstToDouble(*jit_eps),
      broadcast_dims);

  ::llvm::Optional<mlir::Value> affine_weight =
      ctx.GetOptionalMlirValue(node.input(1));
  ::llvm::Optional<mlir::Value> affine_bias =
      ctx.GetOptionalMlirValue(node.input(2));
  ctx.value_map[node.output(0)] = BuildElemAffine(
      builder, loc, norm_input, affine_weight, affine_bias, broadcast_dims);
  return true;
}

namespace {
auto mhlo_conversion = MhloConversionPatternRegister().pattern(
    "aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor",
    ConvertAtenBatchNorm);
}
} // namespace blade
} // namespace torch

#include <mlir/mhlo/builder/softmax.h>

#include "common_utils/logging.h"
#include "compiler/mlir/converters/impl/prim_constant.h"
#include "compiler/mlir/converters/impl/utils.h"
#include "compiler/mlir/converters/mhlo_converter_register.h"
#include "compiler/mlir/converters/mlir_type_utils.h"

#include <torch/script.h>

namespace torch {
namespace addons {
using namespace mlir::mhlo;

template <bool is_logsoftmax = false>
bool ConvertAtenSoftmax(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  const auto& loc = GetNodeLocation(ctx, node);
  const auto& ml_raw_input = ctx.GetMlirValue(node.input(0));
  auto jit_dim = node.input(1);
  auto jit_dtype = node.input(2);
  static constexpr const char* op_name = "aten::softmax";
  if (!CheckConstAttribute(jit_dim, op_name, "dim")) {
    return false;
  }
  if (!CheckConstAttribute(jit_dtype, op_name, "dtype")) {
    return false;
  }

  auto jit_dim_ival = torch::jit::toIValue(jit_dim);
  mlir_dim_t reduce_dim = -1;
  if (jit_dim_ival && !jit_dim_ival->isNone()) {
    reduce_dim = CastJitConstToInt64(*jit_dim);
  }
  auto& builder = *ctx.builder;
  auto optional_input_casted =
      BuildCastWithJitType(builder, loc, ml_raw_input, jit_dtype);
  if (!optional_input_casted) {
    TORCH_CHECK(jit_dtype != nullptr);
    DLOG(INFO)
        << "Could not convert aten::softmax with invalid parameter: dtype %"
        << jit_dtype->debugName();
    return false;
  }

  auto ml_input = *optional_input_casted;

  ctx.value_map[node.output(0)] =
      BuildSoftmax(builder, loc, ml_input, reduce_dim, is_logsoftmax);
  return true;
}

namespace {
auto mhlo_conversion =
    MhloConversionPatternRegister()
        .pattern(
            "aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor",
            ConvertAtenSoftmax)
        .pattern(
            "aten::log_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor",
            ConvertAtenSoftmax<true>);
} // namespace
} // namespace addons
} // namespace torch

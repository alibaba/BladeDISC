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

#include <mlir/mhlo/builder/reduction.h>

#include "pytorch_blade/common_utils/logging.h"
#include "pytorch_blade/compiler/mlir/converters/impl/prim_constant.h"
#include "pytorch_blade/compiler/mlir/converters/impl/utils.h"
#include "pytorch_blade/compiler/mlir/converters/mhlo_converter_register.h"
#include "pytorch_blade/compiler/mlir/converters/mlir_type_utils.h"

#include <torch/script.h>

namespace torch {
namespace blade {
using namespace mlir::mhlo;

template <typename MathOp, bool IsMean = false>
bool ConvertAtenReduction(
    MhloConversionContext& ctx,
    const mlir::Location& loc,
    const mlir::Value& ml_input_val,
    const torch::jit::Value* jit_dims,
    const torch::jit::Value* jit_keepdim,
    const torch::jit::Value* jit_dtype,
    const torch::jit::Value* jit_output) {
  // We use IsPrimConstant here because some prim constant is not supported.
  // So that they have no counter part mlir::Value.
  bool is_const_dims = IsPrimConstant(jit_dims);
  bool is_const_attrs = CheckConstAttribute(jit_dims, "reduction", "dim") &&
      CheckConstAttribute(jit_keepdim, "reduction", "keepdim") &&
      CheckConstAttribute(jit_dtype, "reduction", "dtype");
  if (!is_const_attrs) {
    return false;
  }

  auto& builder = *ctx.builder;
  auto optional_input_val =
      BuildCastWithJitType(builder, loc, ml_input_val, jit_dtype);
  if (!optional_input_val) {
    TORCH_CHECK(jit_dtype != nullptr);
    LOG(WARNING)
        << "Could not convert reduction with invalid parameter: dtype %"
        << jit_dtype->debugName();
    return false;
  }
  auto input_val = *optional_input_val;
  SmallVec4<mlir_dim_t> reduce_dims;
  mlir_dim_t input_rank = GetRankOfMlirValue(input_val);
  if (jit_dims != nullptr) {
    if (ctx.list_map.find(jit_dims) == ctx.list_map.end()) {
      return false;
    }
    auto ml_dim_sizes = ctx.GetMlirValueList(jit_dims);
    reduce_dims.reserve(ml_dim_sizes.size());
    for (auto dsize : ml_dim_sizes) {
      reduce_dims.push_back(*CastStdConstToI64(dsize));
    }
    reduce_dims = NormalizeDimIndex(reduce_dims, input_rank);
    std::sort(reduce_dims.begin(), reduce_dims.end());
  } else {
    reduce_dims = RangeIndices(0, input_rank);
  }

  bool keepdim = jit_keepdim != nullptr && CastJitConstToBool(*jit_keepdim);
  auto elem_type = GetMlirTensorElemType(input_val);
  auto init_value = BuildReductionInitValue<MathOp>(builder, loc, elem_type);
  auto result = BuildReduction<MathOp, IsMean>(
      builder, loc, init_value, input_val, reduce_dims, keepdim);
  TORCH_CHECK(jit_output != nullptr);
  ctx.value_map[jit_output] = result;
  return true;
}

template <typename MathOp, bool IsMean = false>
bool ConvertAtenReductionWoDims(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  auto loc = GetNodeLocation(ctx, node);
  auto input = ctx.GetMlirValue(node.input(0));
  // the desired data type of returned tensor. If specified,
  // the input tensor is casted to dtype before the operation is performed.
  auto jit_dtype = node.input(1);
  return ConvertAtenReduction<MathOp, IsMean>(
      ctx, loc, input, nullptr, nullptr, jit_dtype, node.output(0));
}

template <typename MathOp, bool IsMean = false>
bool ConvertAtenReductionWithDims(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  auto loc = GetNodeLocation(ctx, node);
  auto ml_input_val = ctx.GetMlirValue(node.input(0));
  auto jit_dims = node.input(1);
  auto jit_keepdims = node.input(2);
  auto jit_dtype = node.input(3);
  return ConvertAtenReduction<MathOp, IsMean>(
      ctx,
      loc,
      ml_input_val,
      jit_dims,
      jit_keepdims,
      jit_dtype,
      node.output(0));
  return true;
}

namespace {
auto mhlo_conversion =
    MhloConversionPatternRegister()
        .pattern(
            "aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor",
            ConvertAtenReductionWoDims<mlir::mhlo::AddOp>)
        .pattern(
            R"SIG(aten::sum.dim_IntList(
            Tensor self, int[1] dim, bool keepdim=False, *,
            ScalarType? dtype=None) -> Tensor)SIG",
            ConvertAtenReductionWithDims<mlir::mhlo::AddOp>)
        .pattern(
            "aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor",
            ConvertAtenReductionWoDims<mlir::mhlo::AddOp, true>)
        .pattern(
            R"SIG(aten::mean.dim(
            Tensor self, int[1] dim, bool keepdim=False, *,
            ScalarType? dtype=None) -> Tensor)SIG",
            ConvertAtenReductionWithDims<mlir::mhlo::AddOp, true>);
} // namespace
} // namespace blade
} // namespace torch

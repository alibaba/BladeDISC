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
#include <mlir/mhlo/builder/broadcast.h>
#include <mlir/mhlo/builder/constant.h>
#include <mlir/mhlo/builder/mlir_shape_builder.h>
#include <mlir/mhlo/builder/mlir_utils.h>
#include <mlir/mhlo/builder/standard.h>

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "pytorch_blade/compiler/mlir/converters/mhlo_conversion_context.h"
#include "pytorch_blade/compiler/mlir/converters/mhlo_converter_register.h"
#include "pytorch_blade/compiler/mlir/converters/mlir_type_utils.h"

// was included last because of llvm headers conflicts
#include <torch/script.h>

namespace torch {
namespace blade {
using namespace mlir::mhlo;

bool ConvertAtenZeros(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  const auto& loc = GetNodeLocation(ctx, node);
  auto jit_sizes = node.input(0);
  if (ctx.list_map.find(jit_sizes) == ctx.list_map.end()) {
    return false;
  }

  auto& builder = *ctx.builder;
  auto ml_sizes =
      BuildStdScalarToHloDimType(builder, loc, ctx.GetMlirValueList(jit_sizes));
  auto result_type = BuildMlirRankedTensorType(builder, *node.output(0));
  auto zero =
      BuildHloConstZeroForType(builder, loc, result_type.getElementType());
  ctx.value_map[node.output(0)] =
      BuildBroadcastTensorInDims(builder, loc, zero, ml_sizes, {});

  return true;
}

bool ConvertAtenOnes(MhloConversionContext& ctx, const torch::jit::Node& node) {
  const auto& loc = GetNodeLocation(ctx, node);
  auto jit_sizes = node.input(0);
  if (ctx.list_map.find(jit_sizes) == ctx.list_map.end()) {
    return false;
  }

  auto& builder = *ctx.builder;
  auto ml_sizes =
      BuildStdScalarToHloDimType(builder, loc, ctx.GetMlirValueList(jit_sizes));
  auto result_type = BuildMlirRankedTensorType(builder, *node.output(0));
  auto one =
      BuildHloConstOneForType(builder, loc, result_type.getElementType());
  ctx.value_map[node.output(0)] =
      BuildBroadcastTensorInDims(builder, loc, one, ml_sizes, {});

  return true;
}

// Ref: https://pytorch.org/docs/stable/generated/torch.Tensor.repeat.html
//
// Convert aten::repeat to HLO BroadcastInDim and Reshape ops.
//   For shape [S1, S2, S3] and multiplies [M1, M2, M3]
//     MS1 = M1 * S1; MS2 = M2 * S2; MS3 = M3 * S3
//
// %broadcast = mhlo.broadcast_in_dim(%input) {
//    broadcast_dimensions = [0, 2, 4]
// }
// %result = "mhlo.reshape"(%broadcast) : (tensor<S1xM1xS2xM2xS3xM3xf32>)
//   -> tenosr<MS1xMS2xMS3xf32>
bool ConvertAtenRepeat(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  const auto& loc = GetNodeLocation(ctx, node);
  const auto& ml_input = ctx.GetMlirValue(node.input(0));
  auto jit_repeats = node.input(1);
  if (ctx.list_map.find(jit_repeats) == ctx.list_map.end()) {
    return false;
  }

  auto& builder = *ctx.builder;
  auto inp_dim_sizes = BuildDimSizeListOfTensor(builder, loc, ml_input);
  auto ml_repeats = BuildStdScalarToHloDimType(
      builder, loc, ctx.GetMlirValueList(jit_repeats));
  mlir_dim_t rank = inp_dim_sizes.size();
  mlir_dim_t new_rank = ml_repeats.size();
  mlir_dim_t leading_rank = new_rank - rank;
  TORCH_CHECK(new_rank >= rank);

  SmallVec4<mlir::Value> output_dim_sizes;
  SmallVec4<mlir::Value> broadcast_dim_sizes;
  SmallVec4<mlir_dim_t> broadcast_dims;

  std::copy(
      ml_repeats.begin(),
      ml_repeats.begin() + leading_rank,
      std::back_inserter(output_dim_sizes));
  std::copy(
      ml_repeats.begin(),
      ml_repeats.begin() + leading_rank,
      std::back_inserter(broadcast_dim_sizes));
  for (mlir_dim_t d = leading_rank; d < new_rank; ++d) {
    auto input_dimsz = inp_dim_sizes[d - leading_rank];
    auto repeat_dimsz = ml_repeats[d];
    broadcast_dim_sizes.push_back(repeat_dimsz);
    broadcast_dim_sizes.push_back(input_dimsz);
    output_dim_sizes.push_back(
        builder.create<mlir::arith::MulIOp>(loc, input_dimsz, repeat_dimsz));
    broadcast_dims.push_back(d * 2 - leading_rank + 1);
  }
  auto repeat_tensor = BuildBroadcastTensorInDims(
      builder, loc, ml_input, broadcast_dim_sizes, broadcast_dims);

  ctx.value_map[node.output(0)] =
      BuildDynamicReshapeTensor(builder, loc, repeat_tensor, output_dim_sizes);
  return true;
}

bool ConvertAtenExpand(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  const auto& loc = GetNodeLocation(ctx, node);
  const auto& ml_input = ctx.GetMlirValue(node.input(0));
  auto jit_sizes = node.input(1);
  if (ctx.list_map.find(jit_sizes) == ctx.list_map.end()) {
    return false;
  }
  auto dim_sizes = ctx.GetMlirValueList(jit_sizes);
  mlir_dim_t rank = GetRankOfMlirValue(ml_input);
  mlir_dim_t new_rank = dim_sizes.size();
  TORCH_CHECK(new_rank >= rank);
  mlir_dim_t leading_rank = new_rank - rank;

  auto& builder = *ctx.builder;
  for (mlir_dim_t d = 0; d < new_rank; ++d) {
    if (auto dsize = CastStdConstToI64(dim_sizes[d])) {
      if (*dsize == -1) {
        // Passing -1 as the size for a dimension means not changing the size of
        // that dimension.
        if (d < leading_rank) {
          // For the new leading dimensions, the size cannot be set to -1.
          return false;
        } else {
          // Passing -1 as the size for a dimension means not changing the size
          // of that dimension.
          auto dim_index = d - leading_rank;
          dim_sizes[d] =
              BuildStdDimSizeOfTensor(builder, loc, ml_input, dim_index);
        }
      } else if (*dsize < 0) {
        // illegal expanded dimension size
        return false;
      }
    }
  }
  dim_sizes = BuildStdScalarToHloDimType(builder, loc, dim_sizes);
  auto result = BuildBroadcastTensorInDims(
      builder, loc, ml_input, dim_sizes, RangeIndices(leading_rank, new_rank));
  ctx.value_map[node.output(0)] = result;
  return true;
}

bool ConvertAtenExpandAs(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  const auto& loc = GetNodeLocation(ctx, node);
  const auto& ml_input = ctx.GetMlirValue(node.input(0));
  const auto& ml_other = ctx.GetMlirValue(node.input(1));

  mlir_dim_t rank = GetRankOfMlirValue(ml_input);
  mlir_dim_t new_rank = GetRankOfMlirValue(ml_other);
  if (new_rank < rank) {
    return false;
  }
  mlir_dim_t leading_rank = new_rank - rank;
  auto& builder = *ctx.builder;
  auto dim_sizes = BuildDimSizeListOfTensor(builder, loc, ml_other);
  auto result = BuildBroadcastTensorInDims(
      builder, loc, ml_input, dim_sizes, RangeIndices(leading_rank, new_rank));
  ctx.value_map[node.output(0)] = result;
  return true;
}
namespace {
auto mhlo_conversion =
    MhloConversionPatternRegister()
        .pattern(
            "aten::zeros(int[] size, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)",
            ConvertAtenZeros)
        .pattern(
            "aten::ones(int[] size, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)",
            ConvertAtenOnes)
        .pattern(
            "aten::repeat(Tensor self, int[] repeats) -> (Tensor)",
            ConvertAtenRepeat)
        .pattern(
            // MLIR need SSA, but here expand return the input buffer, the
            // output tensor should be guaranteed not being inplace modified
            // before the converter.
            "aten::expand(Tensor(a) self, int[] size, *, bool implicit=False) -> Tensor(a)",
            ConvertAtenExpand)
        .pattern(
            "aten::expand_as(Tensor self, Tensor other) -> Tensor",
            ConvertAtenExpandAs);
}

} // namespace blade
} // namespace torch

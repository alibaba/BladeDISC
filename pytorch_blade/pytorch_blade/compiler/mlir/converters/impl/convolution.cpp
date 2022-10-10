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

#include <mlir/mhlo/builder/broadcast.h>
#include <mlir/mhlo/builder/convolution.h>
#include <mlir/mhlo/builder/element_wise_binary.h>
#include <mlir/mhlo/builder/mlir_shape_builder.h>
#include <mlir/mhlo/builder/standard.h>

#include "pytorch_blade/compiler/mlir/converters/impl/prim_constant.h"
#include "pytorch_blade/compiler/mlir/converters/impl/utils.h"
#include "pytorch_blade/compiler/mlir/converters/mhlo_converter_register.h"
#include "pytorch_blade/compiler/mlir/converters/mlir_type_utils.h"

#include <torch/script.h>

namespace torch {
namespace blade {
using namespace mlir::mhlo;

bool ConvertAtenConvolution(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  const char* op_name = node.kind().toDisplayString();
  auto jit_weight = node.input(1);
  if (!CheckConstAttribute(jit_weight, op_name, "weight")) {
    return false;
  }
  auto jit_bias = node.input(2);
  if (!CheckConstAttribute(jit_bias, op_name, "bias")) {
    return false;
  }
  auto jit_stride = node.input(3);
  if (!CheckConstAttribute(jit_stride, op_name, "stride")) {
    return false;
  }
#ifdef TORCH_BLADE_BUILD_WITH_CUDA
  // disc-gpu only supports conv1d and conv2d a.t.m
  // TODO(disc): support conv3d on gpu.
  if (CastJitConstListToVec<int64_t>(*jit_stride).size() > 2) {
    return false;
  }
#endif
  auto jit_padding = node.input(4);
  if (!CheckConstAttribute(jit_padding, op_name, "padding")) {
    return false;
  }
  auto jit_dilation = node.input(5);
  if (!CheckConstAttribute(jit_dilation, op_name, "dilation")) {
    return false;
  }
  auto jit_trans = node.input(6);
  if (!CheckConstAttribute(jit_trans, op_name, "transposed")) {
    return false;
  }
  // TODO(disc): support transposed convolution.
  if (CastJitConstToBool(*jit_trans)) {
    return false;
  }
  // TODO(disc): other inputs is ignored currently
  /*
  auto jit_output_pad = node.input(7);
  if (!CheckConstAttribute(jit_output_pad, op_name, "output_padding")) {
    return false;
  }
  auto jit_groups = node.input(8);
  if (!CheckConstAttribute(jit_groups, op_name, "groups")) {
    return false;
  } */

  if (ctx.IsSupportTesting()) {
    return true;
  }

  auto mlir_input = ctx.GetMlirValue(node.input(0));
  auto mlir_weight = ctx.GetMlirValue(jit_weight);
  if (ctx.list_map.find(jit_padding) == ctx.list_map.end()) {
    return false;
  }
  auto mlir_padding_list = ctx.GetMlirValueList(jit_padding);

  SmallValueVec4 mlir_padding_pair;
  size_t n_spatial_dims = mlir_padding_list.size();
  mlir_padding_pair.reserve(n_spatial_dims * 2);
  for (size_t k = 0; k < n_spatial_dims; ++k) {
    // low padding
    mlir_padding_pair.push_back(mlir_padding_list[k]);
    // high padding
    mlir_padding_pair.push_back(mlir_padding_list[k]);
  }
  auto& builder = *ctx.builder;
  auto loc = GetNodeLocation(ctx, node);
  auto mlir_padding = BuildStdScalarToHloTensor(
      builder,
      loc,
      BuildStdScalarToHloDimType(builder, loc, mlir_padding_pair));
  mlir::Value result = BuildConvolution(
      builder,
      loc,
      mlir_input,
      mlir_weight,
      mlir_padding,
      CastJitConstListToVec<int64_t>(*jit_stride),
      CastJitConstListToVec<int64_t>(*jit_dilation));

  // optional bias of shape (out_channels)
  auto jit_bias_ival = torch::jit::toIValue(jit_bias);
  if (jit_bias_ival && !jit_bias_ival->isNone()) {
    auto mlir_bias = ctx.GetMlirValue(jit_bias);
    // broadcast bias to result shape in out_channels_axis
    auto result_dim_sizes = BuildDimSizeListOfTensor(builder, loc, result);
    mlir_dim_t out_channels_axis = 1;
    auto broadcast_bias = BuildBroadcastTensorInDims(
        builder, loc, mlir_bias, result_dim_sizes, {out_channels_axis});
    result = BuildMlirBinaryOp<mlir::chlo::BroadcastAddOp>(
        builder, loc, result, broadcast_bias, GetMlirTensorElemType(result));
  }
  ctx.value_map[node.output(0)] = result;
  return true;
}

namespace {
auto mhlo_conversion = MhloConversionPatternRegister()
                           .pattern(
                               R"SIG(aten::_convolution.deprecated(
                                     Tensor input, Tensor weight, Tensor? bias,
                                     int[] stride, int[] padding, int[] dilation,
                                     bool transposed, int[] output_padding,
                                     int groups, bool benchmark, bool deterministic,
                                     bool cudnn_enabled) -> Tensor)SIG",
                               ConvertAtenConvolution)
                           .pattern(
                               R"SIG(aten::_convolution(
                                     Tensor input, Tensor weight, Tensor? bias,
                                     int[] stride, int[] padding, int[] dilation,
                                     bool transposed, int[] output_padding,
                                     int groups, bool benchmark, bool deterministic,
                                     bool cudnn_enabled, bool allow_tf32) -> Tensor)SIG",
                               ConvertAtenConvolution)
                           .pattern(
                               R"SIG(aten::conv1d(
                                     Tensor input, Tensor weight, Tensor? bias=None,
                                     int[1] stride=1, int[1] padding=0,
                                     int[1] dilation=1, int groups=1) -> Tensor)SIG",
                               ConvertAtenConvolution)
                           .pattern(
                               R"SIG(aten::conv2d(
                                     Tensor input, Tensor weight, Tensor? bias=None,
                                     int[2] stride=1, int[2] padding=0,
                                     int[2] dilation=1, int groups=1) -> Tensor)SIG",
                               ConvertAtenConvolution)
                           .pattern(
                               R"SIG(aten::conv3d(
                                     Tensor input, Tensor weight, Tensor? bias=None,
                                     int[3] stride=1, int[3] padding=0,
                                     int[3] dilation=1, int groups=1) -> Tensor)SIG",
                               ConvertAtenConvolution);
}
} // namespace blade
} // namespace torch

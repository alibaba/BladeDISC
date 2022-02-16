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

#include "mlir/mhlo/builder/algebra_statistics.h"

#include <mlir/mhlo/builder/broadcast.h>
#include <mlir/mhlo/builder/element_wise_binary.h>
#include <mlir/mhlo/builder/reduction.h>

namespace mlir {
namespace mhlo {

mlir::Value BuildStandardNorm(mlir::OpBuilder& builder,
                              const mlir::Location& loc,
                              const mlir::Value& input,
                              const llvm::Optional<mlir::Value>& var,
                              const llvm::Optional<mlir::Value>& mean,
                              double eps,
                              const SmallVec4<mlir_dim_t>& broadcast_dims) {
  auto input_shape = BuildDimSizeListOfTensor(builder, loc, input);
  auto elem_type = GetMlirTensorElemType(input);
  mlir::Value zero_mean = input;
  if (mean) {
    auto b_mean_input = BuildBroadcastTensorInDims(builder, loc, *mean,
                                                   input_shape, broadcast_dims);
    // math: x - broadcast(mean(x))
    zero_mean = BuildMlirBinaryOp<mlir::chlo::BroadcastSubOp>(
        builder, loc, input, b_mean_input, elem_type,
        /* no_implicit_broadcast */ true);
  }
  if (!(var)) {
    return zero_mean;
  }
  // math: var(x) + eps
  static constexpr const char* kDIRECTION = nullptr;
  static constexpr const bool kRHSIsScalar = true;
  // BuildMlirBinaryOp will implicit broadcast at need
  auto var_biased =
      BuildMlirBinaryOp<mlir::chlo::BroadcastAddOp, kDIRECTION, kRHSIsScalar>(
          builder, loc, *var, BuildStdConstForF64(builder, loc, eps), elem_type,
          /* no_implicit_broadcast */ true);
  // math: rsqrt(var(x) + eps)
  auto rsqrt_var = builder.create<mlir::mhlo::RsqrtOp>(loc, var_biased);
  // broadcast rsqrt(var(x) + eps), we do explicit broadcast because we know it.
  auto b_rsqrt_var = BuildBroadcastTensorInDims(builder, loc, rsqrt_var,
                                                input_shape, broadcast_dims);

  // math: (x - mean(x)) * rsqrt(var(x) + eps)
  auto norm_input = BuildMlirBinaryOp<mlir::chlo::BroadcastMulOp>(
      builder, loc, zero_mean, b_rsqrt_var, elem_type,
      /* no_implicit_broadcast */ true);

  return norm_input;
}

mlir::Value BuildStandardNorm(mlir::OpBuilder& builder,
                              const mlir::Location& loc,
                              const mlir::Value& input, double eps,
                              mlir_dim_t reduced_last_dims) {
  mlir_dim_t input_rank = GetRankOfMlirValue(input);
  mlir_dim_t reduced_rank = input_rank - reduced_last_dims;
  SmallVec4<mlir_dim_t> reduce_dims = RangeIndices(reduced_rank, input_rank);

  auto elem_type = GetMlirTensorElemType(input);
  auto zero_value = BuildHloConstZeroForType(builder, loc, elem_type);
  // math: mean(x)
  auto mean_input = BuildReduction<mlir::mhlo::AddOp, true>(
      builder, loc, zero_value, input, reduce_dims);
  auto input_shape = BuildDimSizeListOfTensor(builder, loc, input);
  // broadcast mean, we do explicit broadcast because we know it.
  auto b_mean_input = BuildBroadcastTensorInDims(
      builder, loc, mean_input, input_shape, RangeIndices(0, reduced_rank));
  // math: x - broadcast(mean(x))
  auto zero_mean = BuildMlirBinaryOp<mlir::chlo::BroadcastSubOp>(
      builder, loc, input, b_mean_input, elem_type,
      /* no_implicit_broadcast */ true);
  // math: (x - mean(x))^2
  auto squared_zero_mean = BuildMlirBinaryOp<mlir::chlo::BroadcastMulOp>(
      builder, loc, zero_mean, zero_mean, elem_type,
      /* no_implicit_broadcast */ true);
  // math: var(x)
  auto var_input = BuildReduction<mlir::mhlo::AddOp, true>(
      builder, loc, zero_value, squared_zero_mean, reduce_dims);

  auto norm_input =
      BuildStandardNorm(builder, loc, zero_mean, var_input, llvm::None, eps,
                        RangeIndices(0, reduced_rank));
  return norm_input;
}

mlir::Value BuildElemAffine(
    mlir::OpBuilder& builder, const mlir::Location& loc,
    const mlir::Value& input, const llvm::Optional<mlir::Value>& gamma,
    const llvm::Optional<mlir::Value>& beta,
    const llvm::Optional<SmallVec4<mlir_dim_t>>& broadcast_dims) {
  auto elem_type = GetMlirTensorElemType(input);
  auto input_shape = BuildDimSizeListOfTensor(builder, loc, input);
  mlir::Value affine_output = input;
  if (gamma) {
    mlir::Value weight = *gamma;
    if (broadcast_dims) {
      weight = BuildBroadcastTensorInDims(builder, loc, *gamma, input_shape,
                                          *broadcast_dims);
    }
    affine_output = BuildMlirBinaryOp<mlir::chlo::BroadcastMulOp>(
        builder, loc, affine_output, weight, elem_type,
        /* no_implicit_broadcast */ true);
  }

  if (beta) {
    mlir::Value bias = *beta;
    if (broadcast_dims) {
      bias = BuildBroadcastTensorInDims(builder, loc, *beta, input_shape,
                                        *broadcast_dims);
    }

    affine_output = BuildMlirBinaryOp<mlir::chlo::BroadcastAddOp>(
        builder, loc, affine_output, bias, elem_type,
        /* no_implicit_broadcast */ true);
  }
  return affine_output;
}

}  // namespace mhlo
}  // namespace mlir

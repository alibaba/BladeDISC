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
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/mhlo/builder/mlir_type_utils.h>
#include <mlir/mhlo/builder/mlir_utils.h>
#include <mlir/mhlo/builder/standard.h>
#include <mlir/Dialect/Tensor/IR/TensorOps.h.inc>

#include "pytorch_blade/common_utils/logging.h"
#include "pytorch_blade/compiler/mlir/converters/impl/prim_constant.h"
#include "pytorch_blade/compiler/mlir/converters/impl/utils.h"
#include "pytorch_blade/compiler/mlir/converters/mhlo_conversion_context.h"
#include "pytorch_blade/compiler/mlir/converters/mhlo_converter_register.h"
#include "pytorch_blade/compiler/mlir/converters/mlir_type_utils.h"

#include <torch/script.h>

namespace torch {
namespace blade {

template <bool FromList = false>
bool ConvertAtenTensor(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  auto loc = GetNodeLocation(ctx, node);
  auto jit_dtype = node.input(1);
  if (!CheckConstAttribute(jit_dtype, "aten::tensor", "dtype")) {
    return false;
  }

  auto& builder = *ctx.builder;
  mlir::Value ml_tensor;
  if (FromList) {
    auto jit_list = node.input(0);
    if (ctx.list_map.find(jit_list) == ctx.list_map.end()) {
      return false;
    }
    mlir::mhlo::SmallValueVec4 std_list = ctx.GetMlirValueList(node.input(0));
    ml_tensor = mlir::mhlo::BuildStdScalarToHloTensor(builder, loc, std_list);

  } else {
    mlir::Value std_scalar = ctx.GetMlirValue(node.input(0));
    ml_tensor = mlir::mhlo::BuildStdScalarToHloTensor(builder, loc, std_scalar);
  }
  auto optional_input_casted =
      BuildCastWithJitType(builder, loc, ml_tensor, jit_dtype);
  if (!optional_input_casted) {
    TORCH_CHECK(jit_dtype != nullptr);
    LOG(WARNING)
        << "Could not convert aten::tensor with invalid parameter: dtype %"
        << jit_dtype->debugName();
    return false;
  }

  ctx.value_map[node.output(0)] = *optional_input_casted;
  return true;
}

bool ConvertAtenCat(MhloConversionContext& ctx, const torch::jit::Node& node) {
  auto loc = GetNodeLocation(ctx, node);
  auto jit_list = node.input(0);
  if (ctx.list_map.find(jit_list) == ctx.list_map.end()) {
    return false;
  }
  auto list_vals = ctx.GetMlirValueList(jit_list);
  auto jit_dim = node.input(1);
  if (!CheckConstAttribute(jit_dim, "aten::cat", "dim")) {
    return false;
  }
  auto dim_index = CastJitConstToInt64(*jit_dim);
  auto builder = *ctx.builder;
  auto ranked_type = BuildMlirRankedTensorType(builder, *node.output());
  dim_index = mlir::mhlo::NormalizeDimIndex(dim_index, ranked_type.getRank());
  auto result = builder.create<mlir::mhlo::ConcatenateOp>(
      loc, ranked_type, list_vals, builder.getI64IntegerAttr(dim_index));
  ctx.value_map[node.output(0)] = result;
  return true;
}

bool ConvertAtenItem(MhloConversionContext& ctx, const torch::jit::Node& node) {
  auto loc = GetNodeLocation(ctx, node);
  auto ml_rank0_tensor = ctx.GetMlirValue(node.input(0));
  auto rank = mlir::mhlo::GetRankOfMlirValue(ml_rank0_tensor);
  TORCH_CHECK(rank == 0);
  auto& builder = *ctx.builder;
  ctx.value_map[node.output(0)] =
      builder.create<mlir::tensor::ExtractOp>(loc, ml_rank0_tensor);
  return true;
}

bool ConvertAtenScalarImplicit(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  auto loc = GetNodeLocation(ctx, node);
  auto scalar = ctx.GetMlirValue(node.input(0));
  auto& builder = *ctx.builder;
  ctx.value_map[node.output(0)] =
      mlir::mhlo::BuildStdScalarFromHloTensor(builder, loc, scalar);
  return true;
}

bool ConvertAtenFloat(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  auto std_scalar = ctx.GetMlirValue(node.input(0));
  auto& builder = *ctx.builder;
  // DO TYPE CAST IF NEED
  if (std_scalar.getType().isSignlessInteger()) {
    auto loc = GetNodeLocation(ctx, node);
    ctx.value_map[node.output(0)] = builder.create<mlir::arith::SIToFPOp>(
        loc, builder.getF64Type(), std_scalar);
  } else {
    // must be float
    ctx.value_map[node.output(0)] = std_scalar;
  }
  return true;
}

bool ConvertAtenInt(MhloConversionContext& ctx, const torch::jit::Node& node) {
  auto std_scalar = ctx.GetMlirValue(node.input(0));
  auto& builder = *ctx.builder;
  // DO TYPE CAST IF NEED
  auto type = std_scalar.getType();
  if (type.dyn_cast<mlir::RankedTensorType>()) {
    auto loc = GetNodeLocation(ctx, node);
    std_scalar =
        mlir::mhlo::BuildStdScalarFromHloTensor(builder, loc, std_scalar);
    type = std_scalar.getType();
  }

  auto bit_width = type.getIntOrFloatBitWidth();
  if (type.isInteger(bit_width)) {
    ctx.value_map[node.output(0)] = std_scalar;
  } else if (type.isIntOrFloat()) {
    auto loc = GetNodeLocation(ctx, node);
    ctx.value_map[node.output(0)] = builder.create<mlir::arith::FPToSIOp>(
        loc, builder.getIntegerType(bit_width), std_scalar);
  } else {
    std::string s;
    ::llvm::raw_string_ostream ss(s);
    type.print(ss);
    TORCH_CHECK(false, "can't cast unknown type to Int:", ss.str());
  }
  return true;
}

bool ConvertPrimNumToTensor(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  if (ctx.IsSupportTesting()) {
    return true;
  }

  auto loc = GetNodeLocation(ctx, node);
  auto std_scalar = ctx.GetMlirValue(node.input(0));
  auto& builder = *ctx.builder;
  ctx.value_map[node.output()] =
      mlir::mhlo::BuildStdScalarToHloTensor(builder, loc, std_scalar);
  return true;
}

namespace {
auto mhlo_conversion =
    MhloConversionPatternRegister()
        .pattern(
            "aten::tensor.int(int t, *, int? dtype=None, Device? device=None, bool requires_grad=False) -> (Tensor)",
            ConvertAtenTensor)
        .pattern(
            "aten::tensor.float(float t, *, int? dtype=None, Device? device=None, bool requires_grad=False) -> (Tensor)",
            ConvertAtenTensor)
        .pattern(
            "aten::tensor(t[] data, *, int? dtype=None, Device? device=None, bool requires_grad=False) -> (Tensor)",
            ConvertAtenTensor<true>)
        .pattern(
            "aten::cat(Tensor[] tensors, int dim=0) -> (Tensor)",
            ConvertAtenCat)
        .pattern("aten::item(Tensor self) -> (Scalar)", ConvertAtenItem)
        .pattern("aten::Float.Scalar(Scalar a) -> (float)", ConvertAtenFloat)
        .pattern("aten::Int.Tensor(Tensor a) -> (int)", ConvertAtenInt)
        .pattern("aten::Int.Scalar(Scalar a) -> (int)", ConvertAtenInt)
        .pattern(
            "prim::NumToTensor.Scalar(Scalar a) -> (Tensor)",
            ConvertPrimNumToTensor)
        .pattern(
            "aten::ScalarImplicit(Tensor a) -> (Scalar)",
            ConvertAtenScalarImplicit);
}
} // namespace blade
} // namespace torch

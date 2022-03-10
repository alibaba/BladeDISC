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

#include <mlir-hlo/Dialect/mhlo/IR/chlo_ops.h> // from tf repo
#include <mlir-hlo/Dialect/mhlo/IR/hlo_ops.h> // from tf repo
#include <mlir/Dialect/StandardOps/IR/Ops.h> // from tf repo
#include <mlir/mhlo/builder/element_wise_binary.h>

#include "common_utils/logging.h"
#include "common_utils/utils.h"
#include "compiler/mlir/converters/impl/prim_constant.h"
#include "compiler/mlir/converters/impl/utils.h"
#include "compiler/mlir/converters/mhlo_converter_register.h"
#include "compiler/mlir/converters/mlir_type_utils.h"

#include <torch/script.h>
#include "compiler/jit/shape_type_spec.h"

namespace torch {
namespace blade {
using namespace mlir::mhlo;

template <class MLIR_BINARY_OP, bool rhs_scalar, bool reverse = false>
bool ConvertAtenBinaryOpCommon(
    MhloConversionContext& ctx,
    const torch::jit::Node& node,
    bool has_alpha) {
  auto loc = GetNodeLocation(ctx, node);
  auto jit_inp0 = node.input(0);
  auto jit_inp1 = node.input(1);
  auto hlo_lhs = ctx.GetMlirValue(jit_inp0);
  auto hlo_rhs = ctx.GetMlirValue(jit_inp1);

  auto& builder = *ctx.builder;

  if (rhs_scalar) {
    hlo_rhs = BuildStdScalarToHloTensor(builder, loc, hlo_rhs);
  }

  if (reverse) {
    std::swap(hlo_lhs, hlo_rhs);
  }

  auto result_type = BuildMlirRankedTensorType(builder, *node.output(0));
  bool no_implicit_broadcast = false;
  if (has_alpha) {
    // scale the hlo_rhs if need
    TORCH_CHECK(node.inputs().size() == 3);
    auto std_alpha = ctx.GetMlirValue(node.input(2));
    // update: other = alpha x other

    hlo_rhs = BuildMlirBinaryOp<mlir::chlo::BroadcastMulOp, nullptr, true>(
        builder, loc, hlo_rhs, std_alpha, result_type.getElementType());
    ctx.value_map[node.output(0)] = BuildMlirBinaryOp<MLIR_BINARY_OP>(
        builder,
        loc,
        hlo_lhs,
        hlo_rhs,
        result_type.getElementType(),
        no_implicit_broadcast);
    return true;
  }

  ctx.value_map[node.output(0)] = BuildMlirBinaryOp<MLIR_BINARY_OP>(
      builder,
      loc,
      hlo_lhs,
      hlo_rhs,
      result_type.getElementType(),
      no_implicit_broadcast);
  return true;
}

template <class MLIR_BINARY_OP, bool rhs_scalar = false, bool reverse = false>
bool ConvertAtenBinaryOpWithAlpha(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  return ConvertAtenBinaryOpCommon<MLIR_BINARY_OP, rhs_scalar, reverse>(
      ctx, node, /*has_alpha*/ true);
}

template <class MLIR_BINARY_OP, bool rhs_scalar = false>
bool ConvertAtenBinaryOp(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  return ConvertAtenBinaryOpCommon<MLIR_BINARY_OP, rhs_scalar>(
      ctx, node, /*has_alpha*/ false);
}

template <const char* DIRECTION, bool rhs_scalar = false>
bool ConvertAtenBinaryCompareOp(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  auto loc = GetNodeLocation(ctx, node);
  auto jit_inp0 = node.input(0);
  auto jit_inp1 = node.input(1);
  auto hlo_lhs = ctx.GetMlirValue(jit_inp0);
  auto hlo_rhs = ctx.GetMlirValue(jit_inp1);

  auto& builder = *ctx.builder;
  bool no_implicit_broadcast = false;

  if (GetTrustTracingShape() && jit_inp0->isCompleteTensor() &&
      jit_inp1->isCompleteTensor()) {
    auto inp_type0 = jit_inp0->type()->cast<torch::TensorType>();
    auto inp_type1 = jit_inp1->type()->cast<torch::TensorType>();

    no_implicit_broadcast = inp_type0->sizes() == inp_type1->sizes();
  }

  auto lhs_elem_type = GetMlirTensorElemType(hlo_lhs);
  if (!rhs_scalar) {
    auto rhs_elem_type = GetMlirTensorElemType(hlo_rhs);
    auto rhs_rank = GetRankOfMlirValue(hlo_rhs);
    // TODO: to support type promotion
    if (rhs_rank != 0 && lhs_elem_type != rhs_elem_type) {
      DLOG(INFO)
          << "Could not convert comparision operation with different element type";
      return false;
    }
  }

  auto cmp_elem_type = lhs_elem_type;
  ctx.value_map[node.output(0)] =
      BuildMlirBinaryOp<mlir::chlo::BroadcastCompareOp, DIRECTION, rhs_scalar>(
          builder, loc, hlo_lhs, hlo_rhs, cmp_elem_type, no_implicit_broadcast);
  return true;
}

bool ConvertAtenMaskedFill(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  const auto& loc = GetNodeLocation(ctx, node);
  const auto& input = ctx.GetMlirValue(node.input(0));
  auto mask = ctx.GetMlirValue(node.input(1));
  auto jit_value = node.input(2);
  auto value = ctx.GetMlirValue(jit_value);

  auto& builder = *ctx.builder;
  auto jit_ival = torch::jit::toIValue(jit_value);
  if (jit_ival && !jit_ival->isTensor()) {
    // the value must be a std::scalar, convert it to mhlo::tensor first
    value = BuildStdScalarToHloTensor(builder, loc, value);
  }

  auto b_mask = BuildBroadcastTensorAsOther(builder, loc, mask, input);
  auto t_value = TryBuildElementTypeCast(
      builder, loc, value, GetMlirTensorElemType(input));
  auto b_value = BuildBroadcastTensorAsOther(builder, loc, t_value, input);
  auto result =
      builder.create<mlir::mhlo::SelectOp>(loc, b_mask, b_value, input);
  ctx.value_map[node.output(0)] = result.getResult();
  return true;
}

template <class StdBinOp>
bool ConvertAtenBinaryScalar(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  const auto& loc = GetNodeLocation(ctx, node);
  auto std_lhs = ctx.GetMlirValue(node.input(0));
  auto std_rhs = ctx.GetMlirValue(node.input(1));
  ctx.value_map[node.output()] =
      ctx.builder->create<StdBinOp>(loc, std_lhs, std_rhs);
  return true;
}

bool ConvertAtenScalarCmpIOp(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  const auto& loc = GetNodeLocation(ctx, node);
  auto std_lhs = ctx.GetMlirValue(node.input(0));
  auto std_rhs = ctx.GetMlirValue(node.input(1));
  ctx.value_map[node.output()] = ctx.builder->create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, std_lhs, std_rhs);
  return true;
}

// This converter fails at op lowering phase sometimes. Mainly because omitted
// dtype argument. The plan to enhance this converter is as following:
//   1. Get target dtype;
//   2. If `end` is not integer type (i.e., float type, we only deal with float
//      and integer), assign ceil(end) to end;
//   3. Build iota-op according.
//   4. Convert iota-op to the target dtype.
// The difficulty now is how to get the global default datatype if dtype is
// empty at step 1. Rewrite this converter when figuring out the solution.

// A workaround solution is to put aten::arange to black list:
//   config.customize_op_black_list = ['aten::arange']
bool ConvertAtenArange(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  const auto& loc = GetNodeLocation(ctx, node);
  auto end = ctx.GetMlirValue(node.input(0));
  auto mlir_scalar = ctx.GetMlirValue(node.input(1));

  mlir_scalar.dump();
  std::cout << "GTY0 :" << std::endl;
  auto dtype_value = CastStdConstToI64(mlir_scalar);
  if (!dtype_value) {
    return false;
  }
  std::cout << "GTY1 :" << *dtype_value << std::endl;
  // ScalarType in jit::Graph is type of int
  torch::ScalarType dtype = static_cast<torch::ScalarType>(*dtype_value);
  std::cout << "GTY1 :" << ScalarTypeToString(dtype) << std::endl;

  auto& builder = *ctx.builder;
  std::vector<mlir::Value> dim_sizes = {end};
  end = builder.create<mlir::tensor::FromElementsOp>(loc, dim_sizes);
  auto elem_type = BuildMlirElemType(builder, dtype);
  end = TryBuildElementTypeCast(builder, loc, end, elem_type);

  std::vector<mlir_dim_t> out_shape_vec(1, mlir::ShapedType::kDynamicSize);
  auto out_shape = mlir::RankedTensorType::get(out_shape_vec, elem_type);
  mlir::Value out = builder.create<mlir::mhlo::DynamicIotaOp>(
      loc, out_shape, end, ctx.builder->getI64IntegerAttr(0));
  ctx.value_map[node.output()] = out;
  return true;
}

bool ConvertAtenArangeV2(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  const auto& loc = GetNodeLocation(ctx, node);
  auto dtype = node.input(1);
  if (!IsPrimConstant(*dtype)) {
    DLOG(INFO) << "Only support static dtype.";
    return false;
  }
  auto end = ctx.GetMlirValue(node.input(0));
  auto input_mlir_type = end.getType();
  auto target_dtype = torch::jit::toIValue(node.input(1));

  // See document: https://pytorch.org/docs/stable/generated/torch.arange.html
  // if dtype is None, use a global default. If dtype is not given, infer the
  // data type from the other input arguments. If any of start, end, or stop are
  // floating-point, the dtype is inferred to be the default dtype. Otherwise,
  // the dtype is inferred to be torch.int64.
  c10::ScalarType default_scalar_type =
      c10::typeMetaToScalarType(c10::get_default_dtype());
  c10::ScalarType target_scalar_type;
  if (!target_dtype) {
#if 1
    llvm::errs() << "[ZZ] target-dtype is empty now.\n";
#endif
    target_scalar_type = default_scalar_type;
    // target_dtype = default_dtype;
  } else if (target_dtype->isNone()) {
#if 1
    llvm::errs() << "[ZZ] target_dtype is none.\n";
#endif
    // Do not know how to deal with a datatype that is neither int/index nor
    // float.
    if (!input_mlir_type.isIntOrIndexOrFloat()) {
      DLOG(INFO) << "Unsupported dtype of argument `end` for arange.";
      return false;
    }
    if (input_mlir_type.isBF16() || input_mlir_type.isF16() ||
        input_mlir_type.isF32() || input_mlir_type.isF64()) {
      // target_dtype = default_dtype;
      target_scalar_type = default_scalar_type;
#if 1
      llvm::errs() << "[ZZ] input is float.\n";
#endif
    } else {
      // target_dtype = IValue(static_cast<int64_t>(torch::ScalarType::Long));
      target_scalar_type = torch::ScalarType::Long;
    }
  } else {
    target_scalar_type = static_cast<torch::ScalarType>(target_dtype->toInt());
#if 1
    llvm::errs() << "[ZZ] target_dtype is the same with specified.\n";
#endif
  }

  // If `end` is not int type, convert to int to meet the requirement of
  // mlir::mhlo::DynamicIotaOp.
  auto iota_type = input_mlir_type;
  if (!input_mlir_type.isIntOrIndex()) {
    // First, ceil up `end`, e.g., 3.5 -> 4.0.
    end = ctx.builder->create<mlir::mhlo::CeilOp>(loc, end);
    // Convert type to int.
    auto mlir_long_type =
        BuildMlirElemType(*ctx.builder, c10::ScalarType::Long);
    end = ctx.builder->create<mlir::mhlo::ConvertOp>(loc, end, mlir_long_type);
    iota_type = end.getType();
  }

  std::vector<mlir::Value> dim_sizes = {end};
  end = ctx.builder->create<mlir::tensor::FromElementsOp>(loc, dim_sizes);

  std::vector<mlir_dim_t> out_shape_vec(1, mlir::ShapedType::kDynamicSize);
  auto out_shape = mlir::RankedTensorType::get(out_shape_vec, iota_type);
  mlir::Value iota = ctx.builder->create<mlir::mhlo::DynamicIotaOp>(
      loc, out_shape, end, ctx.builder->getI64IntegerAttr(0));

  // Convert result to target type if necessary.
  auto mlir_target_dtype = BuildMlirElemType(*ctx.builder, target_scalar_type);
#if 1
  llvm::errs() << "[ZZ] mlir-target-dtype is " << mlir_target_dtype << "\n";
#endif
  mlir::Value result;
  if (mlir_target_dtype != iota_type) {
    result = ctx.builder->create<mlir::mhlo::ConvertOp>(
        loc, iota, mlir_target_dtype);
  } else {
    result = iota;
  }

  ctx.value_map[node.output()] = result;
  return true;
}

template <bool rhs_scalar = false>
bool ConvertAtenFloorDiv(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  const auto& loc = GetNodeLocation(ctx, node);
  auto hlo_lhs = ctx.GetMlirValue(node.input(0));
  auto hlo_rhs = ctx.GetMlirValue(node.input(1));

  // According to the document, it actually rounds the quotient towards zero
  // instead of taking its floor. If dtype is integer, it is equivalent to div.
  // Otherwise we process it as following currently:
  //    div = lhs / rhs
  //    a = sign(div)
  //    b = abs(div)
  //    c = floor(b)
  //    result = mul(a, c)

  mlir::Value result;
  auto& builder = *ctx.builder;
  mlir::Value divide =
      BuildMlirBinaryOp<mlir::chlo::BroadcastDivOp, nullptr, rhs_scalar>(
          builder,
          loc,
          hlo_lhs,
          hlo_rhs,
          BuildMlirRankedTensorType(builder, *node.output(0)).getElementType(),
          /* no_implicit_broadcast */ false);

  auto tensor_type = node.output(0)->type()->cast<TensorType>();
  TORCH_CHECK(tensor_type != nullptr);
  if (c10::isIntegralType(
          *(tensor_type->scalarType()),
          /* includeBool */ true)) {
    result = divide;
  } else {
    mlir::Value sign = ctx.builder->create<mlir::mhlo::SignOp>(loc, divide);
    mlir::Value abs = ctx.builder->create<mlir::mhlo::AbsOp>(loc, divide);
    mlir::Value floor = ctx.builder->create<mlir::mhlo::FloorOp>(loc, abs);
    result = ctx.builder->create<mlir::mhlo::MulOp>(loc, sign, floor);
  }
  ctx.value_map[node.output(0)] = result;

  return true;
}

namespace {
static constexpr const char kCompare_GT[] = "GT";
static constexpr const char kCompare_GE[] = "GE";
static constexpr const char kCompare_EQ[] = "EQ";
static constexpr const char kCompare_NE[] = "NE";
static constexpr const char kCompare_LE[] = "LE";
static constexpr const char kCompare_LT[] = "LT";

auto mhlo_conversion =
    MhloConversionPatternRegister()
        .pattern(
            "aten::__and__.bool(bool a, bool b) -> (bool)",
            ConvertAtenBinaryScalar<mlir::arith::AndIOp>)
        .pattern(
            "aten::add.int(int a, int b) -> (int)",
            ConvertAtenBinaryScalar<mlir::arith::AddIOp>)
        .pattern(
            "aten::add.Tensor(Tensor self, Tensor other, Scalar alpha=1) "
            "-> Tensor",
            ConvertAtenBinaryOpWithAlpha<mlir::chlo::BroadcastAddOp>)
        .pattern(
            "aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) "
            "-> Tensor",
            ConvertAtenBinaryOpWithAlpha<mlir::chlo::BroadcastAddOp, true>)
        .pattern(
            "aten::sub.int(int a, int b) -> (int)",
            ConvertAtenBinaryScalar<mlir::arith::SubIOp>)
        .pattern(
            "aten::sub.Tensor(Tensor self, Tensor other, Scalar alpha=1) "
            "-> Tensor",
            ConvertAtenBinaryOpWithAlpha<mlir::chlo::BroadcastSubOp>)
        .pattern(
            "aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) "
            "-> Tensor",
            ConvertAtenBinaryOpWithAlpha<mlir::chlo::BroadcastSubOp, true>)
        .pattern(
            "aten::mul.int(int a, int b) -> (int)",
            ConvertAtenBinaryScalar<mlir::arith::MulIOp>)
        .pattern(
            "aten::mul.Tensor(Tensor self, Tensor other) -> Tensor",
            ConvertAtenBinaryOp<mlir::chlo::BroadcastMulOp>)
        .pattern(
            "aten::mul.Scalar(Tensor self, Scalar other) -> Tensor",
            ConvertAtenBinaryOp<mlir::chlo::BroadcastMulOp, true>)
        .pattern(
            "aten::div.int(int a, int b) -> (int)",
            ConvertAtenBinaryScalar<mlir::arith::DivSIOp>)
        .pattern(
            "aten::div.Tensor(Tensor self, Tensor other) -> Tensor",
            ConvertAtenBinaryOp<mlir::chlo::BroadcastDivOp>)
        .pattern(
            "aten::div.Scalar(Tensor self, Scalar other) -> Tensor",
            ConvertAtenBinaryOp<mlir::chlo::BroadcastDivOp, true>)
        .pattern(
            "aten::floor_divide(Tensor self, Tensor other) -> Tensor",
            ConvertAtenFloorDiv)
        .pattern(
            "aten::floor_divide.Scalar(Tensor self, Scalar other) -> Tensor",
            ConvertAtenFloorDiv<true>)
        // Ref: https://pytorch.org/docs/stable/generated/torch.true_divide.html
        // torch.true_divide, alias for torch.div
        .pattern(
            "aten::true_divide.Tensor(Tensor self, Tensor other) -> Tensor",
            ConvertAtenBinaryOp<mlir::chlo::BroadcastDivOp>)
        .pattern(
            "aten::true_divide.Scalar(Tensor self, Scalar other) -> Tensor",
            ConvertAtenBinaryOp<mlir::chlo::BroadcastDivOp, true>)
        // binary comparision ops
        .pattern(
            "aten::gt.Tensor(Tensor self, Tensor other) -> Tensor",
            ConvertAtenBinaryCompareOp<kCompare_GT>)
        .pattern(
            "aten::gt.Scalar(Tensor self, Scalar other) -> Tensor",
            ConvertAtenBinaryCompareOp<kCompare_GT, true>)
        .pattern(
            "aten::ge.Tensor(Tensor self, Tensor other) -> Tensor",
            ConvertAtenBinaryCompareOp<kCompare_GE>)
        .pattern(
            "aten::ge.Scalar(Tensor self, Scalar other) -> Tensor",
            ConvertAtenBinaryCompareOp<kCompare_GE, true>)
        .pattern(
            "aten::eq.Tensor(Tensor self, Tensor other) -> Tensor",
            ConvertAtenBinaryCompareOp<kCompare_EQ>)
        .pattern(
            "aten::eq.Scalar(Tensor self, Scalar other) -> Tensor",
            ConvertAtenBinaryCompareOp<kCompare_EQ, true>)
        .pattern(
            "aten::ne.Tensor(Tensor self, Tensor other) -> Tensor",
            ConvertAtenBinaryCompareOp<kCompare_NE>)
        .pattern(
            "aten::ne.Scalar(Tensor self, Scalar other) -> Tensor",
            ConvertAtenBinaryCompareOp<kCompare_NE, true>)
        .pattern(
            "aten::le.Tensor(Tensor self, Tensor other) -> Tensor",
            ConvertAtenBinaryCompareOp<kCompare_LE>)
        .pattern(
            "aten::le.Scalar(Tensor self, Scalar other) -> Tensor",
            ConvertAtenBinaryCompareOp<kCompare_LE, true>)
        .pattern(
            "aten::lt.Tensor(Tensor self, Tensor other) -> Tensor",
            ConvertAtenBinaryCompareOp<kCompare_LT>)
        .pattern(
            "aten::lt.Scalar(Tensor self, Scalar other) -> Tensor",
            ConvertAtenBinaryCompareOp<kCompare_LT, true>)
        .pattern(
            "aten::lt.int(int a, int b) -> (bool)",
            ConvertAtenScalarCmpIOp)
        .pattern(
            "aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> (Tensor)",
            ConvertAtenBinaryOp<mlir::chlo::BroadcastPowOp>)
        .pattern(
            "aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> (Tensor)",
            ConvertAtenBinaryOp<mlir::chlo::BroadcastPowOp, true>)
        .pattern(
            "aten::masked_fill.Scalar(Tensor self, Tensor mask, Scalar "
            "value) -> Tensor",
            ConvertAtenMaskedFill)
        .pattern(
            "aten::masked_fill.Tensor(Tensor self, Tensor mask, Tensor "
            "value) -> Tensor",
            ConvertAtenMaskedFill)
        .pattern(
            "aten::rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) "
            "-> (Tensor)",
            ConvertAtenBinaryOpWithAlpha<
                mlir::chlo::BroadcastSubOp,
                true,
                true>)
        .pattern(
            "aten::rsub.Tensor(Tensor self, Tensor other, Scalar alpha=1) -> "
            "(Tensor)",
            ConvertAtenBinaryOpWithAlpha<
                mlir::chlo::BroadcastSubOp,
                false,
                true>)
        .pattern(
            "aten::arange(Scalar end, int? dtype=None, int? layout=None, "
            "Device? device=None, bool? pin_memory=None) -> (Tensor)",
            ConvertAtenArange);
// ConvertAtenArangeV2);
} // namespace
} // namespace blade
} // namespace torch

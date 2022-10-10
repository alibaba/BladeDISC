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

#include "pytorch_blade/compiler/mlir/converters/impl/prim_constant.h"

#include <mlir/mhlo/builder/constant.h>
#include <mlir/mhlo/builder/mlir_attr_utils.h>
#include <mlir/mhlo/builder/standard.h>

#include "pytorch_blade/compiler/mlir/converters/mhlo_conversion_context.h"
#include "pytorch_blade/compiler/mlir/converters/mhlo_converter_register.h"
#include "pytorch_blade/compiler/mlir/converters/mlir_type_utils.h"

// was included last because of llvm headers conflicts
#include <torch/script.h>

namespace torch {
namespace blade {
using namespace mlir::mhlo;

bool IsPrimConstant(const torch::jit::Value& value) {
  return value.node()->kind() == prim::Constant;
}

bool IsPrimConstant(const torch::jit::Value* val) {
  if (val == nullptr) {
    return true;
  }
  return IsPrimConstant(*val);
}

template <>
std::vector<int64_t> CastJitConstListToVec<int64_t>(
    const torch::jit::Value& jit_val) {
  auto const_ival = torch::jit::toIValue(&jit_val);
  TORCH_CHECK(
      const_ival && const_ival->isIntList(),
      "The input torch::jit::Value must be Int List");
  return const_ival->toIntList().vec();
}

template <>
std::vector<double> CastJitConstListToVec<double>(
    const torch::jit::Value& jit_val) {
  auto const_ival = torch::jit::toIValue(&jit_val);
  TORCH_CHECK(
      const_ival && const_ival->isDoubleList(),
      "The input torch::jit::Value must be Double List");
  return const_ival->toDoubleList().vec();
}

template <typename T>
T CastJitConstToNumeric(const torch::jit::Value& val) {
  TORCH_CHECK(
      IsPrimConstant(val),
      "The torch::jit::Value %",
      val.debugName(),
      " producer node must be prim::Constant");
  auto const_ival = torch::jit::toIValue(&val);
  TORCH_CHECK(
      const_ival, "The torch::jit::Value %", val.debugName(), " is empty");
  if (const_ival->isInt()) {
    if (!std::is_same<T, int64_t>::value) {
      LOG(WARNING) << "An int64_t constant was cast to type:"
                   << typeid(T).name();
    }
    return const_ival->toInt();
  } else if (const_ival->isDouble()) {
    if (!std::is_same<T, double>::value) {
      LOG(WARNING) << "A double constant was cast to type:" << typeid(T).name();
    }
    return const_ival->toDouble();
  } else if (const_ival->isBool()) {
    if (!std::is_same<T, bool>::value) {
      LOG(WARNING) << "A bool constant was cast to type:" << typeid(T).name();
    }
    return const_ival->toBool();
  }

  TORCH_CHECK(
      false,
      "The torch::jit::Value %",
      val.debugName(),
      " can't be cast to type:",
      typeid(T).name());
}

std::string CastJitConstToString(const torch::jit::Value& val) {
  TORCH_CHECK(
      IsPrimConstant(val),
      "The torch::jit::Value %",
      val.debugName(),
      " producer node must be prim::Constant");
  auto const_ival = torch::jit::toIValue(&val);
  TORCH_CHECK(
      const_ival, "The torch::jit::Value %", val.debugName(), " is empty");
  return const_ival->toString()->string();
}

int64_t CastJitConstToInt64(const torch::jit::Value& val) {
  return CastJitConstToNumeric<int64_t>(val);
}

double CastJitConstToDouble(const torch::jit::Value& val) {
  return CastJitConstToNumeric<double>(val);
}

bool CastJitConstToBool(const torch::jit::Value& val) {
  return CastJitConstToNumeric<bool>(val);
}

mlir::Value BuildMlirConstFromTorchTensor(
    mlir::OpBuilder& builder,
    const mlir::Location& loc,
    const at::Tensor& val) {
  const auto& sizes = val.sizes();
  std::vector<mlir_dim_t> ranked_shape(sizes.begin(), sizes.end());
  auto scalar_type = val.scalar_type();
  auto elem_type = BuildMlirElemType(builder, scalar_type);
  auto ranked_tensor_type =
      mlir::RankedTensorType::get(ranked_shape, elem_type);
  auto raw_tensor = val.cpu().contiguous();
  mlir::DenseElementsAttr attr;
  if (raw_tensor.scalar_type() == torch::kBool) {
    // at::Tensor store bool in bytes,
    // and ElementBoolAttr store bool in bits.
    ::llvm::SmallVector<bool, 8> bool_buffer;
    std::copy(
        raw_tensor.data_ptr<bool>(),
        raw_tensor.data_ptr<bool>() + raw_tensor.numel(),
        std::back_inserter(bool_buffer));
    attr = BuildElementsAttr<bool>(builder, bool_buffer);
  } else {
    ::llvm::ArrayRef<char> raw_buffer(
        (char*)raw_tensor.data_ptr(), raw_tensor.nbytes());
    attr = mlir::DenseElementsAttr::getFromRawBuffer(
        ranked_tensor_type, raw_buffer);
  }
  auto result = builder.create<mlir::mhlo::ConstantOp>(loc, attr);
  return result.getResult();
}

struct PrimConstantConverter {
  PrimConstantConverter(bool try_flag) : try_support(try_flag) {}
  bool Convert(MhloConversionContext& ctx, const torch::jit::Node& node);

  template <typename T>
  mlir::Value ConvertNumeric(
      MhloConversionContext& ctx,
      const mlir::Location& loc,
      const torch::jit::Node& node,
      T val) {
    auto& builder = *ctx.builder;
    mlir::Value result;
    if (std::is_same<T, int64_t>::value) {
      result = BuildStdConstForI64(builder, loc, val);
    } else if (std::is_same<T, double>::value) {
      result = BuildStdConstForF64(builder, loc, val);
    } else if (std::is_same<T, bool>::value) {
      result = BuildStdConstForBool(builder, loc, val);
    } else {
      TORCH_CHECK(false, "numeric type:", typeid(T).name(), " is unsupported");
    }
    return result;
  }

  template <typename T>
  bool ConvertNumericList(
      MhloConversionContext& ctx,
      const mlir::Location& loc,
      const torch::jit::Node& node,
      const mlir::ArrayRef<T>& vals) {
    SmallVec4<mlir::Value> const_vals;
    for (auto v : vals) {
      const_vals.push_back(ConvertNumeric<T>(ctx, loc, node, v));
    }
    ctx.list_map[node.output(0)] = const_vals;
    return true;
  }

  bool ConvertTensor(
      MhloConversionContext& ctx,
      mlir::Location loc,
      const torch::jit::Node& node,
      at::Tensor val) {
    auto& builder = *ctx.builder;
    ctx.value_map[node.output(0)] =
        BuildMlirConstFromTorchTensor(builder, loc, val);
    return true;
  }

  bool try_support;
};

bool PrimConstantConverter::Convert(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  auto loc = GetNodeLocation(ctx, node);
  auto const_ival = torch::jit::toIValue(node.output());
  if (!const_ival)
    return false;

  auto val = *const_ival;
  if (val.isTensor()) {
    at::Tensor ref = val.toTensor();
    return try_support ? true : ConvertTensor(ctx, loc, node, ref);
  } else if (val.isInt()) {
    ctx.value_map[node.output(0)] =
        ConvertNumeric<int64_t>(ctx, loc, node, val.toInt());
    return true;
  } else if (val.isDouble()) {
    ctx.value_map[node.output(0)] =
        ConvertNumeric<double>(ctx, loc, node, val.toDouble());
    return true;
  } else if (val.isBool()) {
    ctx.value_map[node.output(0)] =
        ConvertNumeric<bool>(ctx, loc, node, val.toBool());
    return true;
  } else if (val.isList()) {
    if (val.isIntList()) {
      std::vector<int64_t> int64_vec = val.toIntList().vec();
      return try_support
          ? true
          : ConvertNumericList<int64_t>(ctx, loc, node, int64_vec);
    } else if (val.isDoubleList()) {
      std::vector<double> double_vec = val.toDoubleList().vec();
      return try_support
          ? true
          : ConvertNumericList<double>(ctx, loc, node, double_vec);
    } else if (val.isBoolList()) {
      ::llvm::SmallVector<bool, 8> bool_vec;
      auto bool_list = val.toBoolList().vec();
      std::copy(
          bool_list.begin(), bool_list.end(), std::back_inserter(bool_vec));
      return try_support ? true
                         : ConvertNumericList<bool>(ctx, loc, node, bool_vec);
    }
  } else if (val.isString()) {
  } else if (val.isDevice()) {
  } else if (val.isNone()) {
  } else if (val.isTuple()) {
  } else if (val.isGenericDict()) {
  } else {
  }

  // prim::Constant would be clustered according to its' usage but not it's
  // support info for example, the following prim::Constant is not support but
  // it's need by aten::sum, so it should be clustered together with aten::sum
  //   %1 : None = prim::Constant()
  //   %2 : Float() = aten::sum(%x_, %1)
  //
  // So that, do nothing and return true if the prim::Constant is not support
  return !try_support;
}

bool ConvertPrimConstant(
    MhloConversionContext& ctx,
    const torch::jit::Node& node) {
  PrimConstantConverter converter(ctx.IsSupportTesting());
  return converter.Convert(ctx, node);
}

namespace {
auto mhlo_conversion = MhloConversionPatternRegister().pattern(
    GetPrimOperatorName(prim::Constant),
    ConvertPrimConstant);
} // namespace
} // namespace blade
} // namespace torch

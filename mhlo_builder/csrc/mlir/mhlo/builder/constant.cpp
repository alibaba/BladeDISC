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

#include "mlir/mhlo/builder/constant.h"

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"

namespace mlir {
namespace mhlo {
bool IsHloConstant(const mlir::Value& value) {
  auto def =
      llvm::dyn_cast_or_null<mlir::mhlo::ConstantOp>(value.getDefiningOp());
  return def != nullptr;
}

bool IsStdConstant(const mlir::Value& value) {
  auto def =
      llvm::dyn_cast_or_null<mlir::arith::ConstantOp>(value.getDefiningOp());
  return def != nullptr;
}

template <typename T>
llvm::Optional<T> CastHloConstToElementsAttr(const mlir::Value& val) {
  auto def = llvm::dyn_cast<mlir::mhlo::ConstantOp>(val.getDefiningOp());
  if (!def) {
    return llvm::None;
  }
  auto const_value = def.value();
  return const_value.dyn_cast_or_null<T>();
}

SmallVec4<int64_t> CastHloConstToListOfI64(const mlir::Value& value) {
  // cast will throw exception if meet error
  auto ml_elem_attr =
      CastHloConstToElementsAttr<mlir::DenseIntElementsAttr>(value);
  MHLO_CHECK(ml_elem_attr, "The input mlir::Value could not cast to const");
  SmallVec4<int64_t> vec_i64;
  // APInt: arbitrary precision integers.
  for (const auto& ap_index : ml_elem_attr->getValues<mlir::APInt>()) {
    int64_t index = ap_index.getSExtValue();
    vec_i64.push_back(index);
  }
  return vec_i64;
}

llvm::Optional<int64_t> CastAttrToI64(const mlir::Attribute& def) {
  auto attr = def.dyn_cast_or_null<mlir::IntegerAttr>();
  if (attr) {
    int64_t index = attr.getValue().getSExtValue();
    return index;
  } else {
    return llvm::None;
  }
}

llvm::Optional<int64_t> CastHloConstToI64(const mlir::Value& val) {
  SmallVec4<int64_t> vec_i64 = CastHloConstToListOfI64(val);
  if (vec_i64.size() == 1) {
    return vec_i64[0];
  } else {
    return llvm::None;
  }
}

llvm::Optional<int64_t> CastStdConstToI64(const mlir::Value& val) {
  auto def = llvm::dyn_cast<mlir::arith::ConstantOp>(val.getDefiningOp());
  if (!def) {
    return llvm::None;
  }
  return CastAttrToI64(def.getValue());
}
}  // namespace mhlo
}  // namespace mlir

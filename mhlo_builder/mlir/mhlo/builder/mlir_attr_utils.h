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

#pragma once
#include "mlir/mhlo/builder/macros.h"
#include "mlir/mhlo/builder/mlir_type_utils.h"

namespace mlir {
namespace mhlo {
// Returns 1D bool dense elements attribute with the given values.
template <typename T>
mlir::DenseElementsAttr BuildElementsAttr(mlir::OpBuilder& builder,
                                          const mlir::ArrayRef<T>& values) {
  mlir::Type elem_type;
  if (std::is_same<T, int64_t>::value) {
    elem_type = builder.getIntegerType(64);
  } else if (std::is_same<T, bool>::value) {
    elem_type = builder.getI1Type();
  } else if (std::is_same<T, double>::value) {
    elem_type = builder.getF64Type();
  } else {
    MHLO_CHECK(false, " is not supported");
  }
  mlir::RankedTensorType ty = mlir::RankedTensorType::get(
      {static_cast<int64_t>(values.size())}, elem_type);
  auto attr = mlir::DenseElementsAttr::get(ty, values);
  return attr;
}

// Returns 1D 64-bit dense elements attribute with the given values.
mlir::DenseIntElementsAttr BuildI64ElementsAttr(
    mlir::OpBuilder& builder, const mlir::ArrayRef<int64_t>& values);
}  // namespace mhlo
}  // namespace mlir

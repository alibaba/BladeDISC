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

#include "mlir/mhlo/builder/mlir_attr_utils.h"

namespace mlir {
namespace mhlo {

// Returns 1D 64-bit dense elements attribute with the given values.
mlir::DenseIntElementsAttr BuildI64ElementsAttr(
    mlir::OpBuilder& builder, const mlir::ArrayRef<int64_t>& values) {
  mlir::RankedTensorType ty = mlir::RankedTensorType::get(
      {static_cast<int64_t>(values.size())}, builder.getIntegerType(64));
  return mlir::DenseIntElementsAttr::get(ty, values);
}
}  // namespace mhlo
}  // namespace mlir
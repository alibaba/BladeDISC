// Copyright 2022 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DISC_TOOLS_DISC_TRANSFORM_LINALGEXT_OPS_EXT_
#define DISC_TOOLS_DISC_TRANSFORM_LINALGEXT_OPS_EXT_

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/disc/tools/disc-transform/LinalgExt/LinalgExtEnums.h.inc"
#include "mlir/disc/tools/disc-transform/LinalgExt/LinalgExtInterfaces.h"

namespace mlir {
namespace disc_ral {
namespace disc_linalg_ext {

using linalg::LinalgOp;

}  // namespace disc_linalg_ext
}  // namespace disc_ral
}  // namespace mlir

#define GET_OP_CLASSES
#include "mlir/disc/tools/disc-transform/LinalgExt/LinalgExtOps.h.inc"

namespace mlir {
namespace disc_ral {
namespace disc_linalg_ext {

// Returns dimension value vector for `shapedTypeValue`.
SmallVector<OpFoldResult> getDims(OpBuilder& builder, Location loc,
                                  Value shapedTypeValue);

/// Returns a vector that interchanges `elements` starting at offset `offset`
/// based on the indexes in `interchangeVector`.
template <typename T>
SmallVector<T> interchange(ArrayRef<T> elements,
                           ArrayRef<int64_t> interchangeVector,
                           int offset = 0) {
  SmallVector<T> vec = llvm::to_vector(elements);
  for (auto en : llvm::enumerate(interchangeVector)) {
    vec[en.index() + offset] = elements[en.value() + offset];
  }
  return vec;
}

void registerTilingInterfaceExternalModels(DialectRegistry& registry);
void registerBufferizableOpInterfaceExternalModels(DialectRegistry& registry);

}  // namespace disc_linalg_ext
}  // namespace disc_ral
}  // namespace mlir

#endif  // DISC_TOOLS_DISC_TRANSFORM_LINALGEXT_OPS_EXT_

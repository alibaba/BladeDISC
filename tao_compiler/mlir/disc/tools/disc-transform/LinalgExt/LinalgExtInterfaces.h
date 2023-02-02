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

#ifndef DISC_TOOLS_DISC_TRANSFORM_LINALGEXT_INTERFACES_EXT_
#define DISC_TOOLS_DISC_TRANSFORM_LINALGEXT_INTERFACES_EXT_

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace disc_ral {
namespace disc_linalg_ext {
class LinalgExtOp;

namespace detail {
LogicalResult verifyLinalgExtOpInterface(Operation* op);
}

/// Include the generated interface declarations.
#include "mlir/disc/tools/disc-transform/LinalgExt/LinalgExtOpInterfaces.h.inc"

}  // namespace disc_linalg_ext
}  // namespace disc_ral
}  // namespace mlir

#endif  // DISC_TOOLS_DISC_TRANSFORM_LINALGEXT_INTERFACES_EXT_
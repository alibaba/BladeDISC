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

#include "mlir/disc/tools/disc-transform/LinalgExt/LinalgExtDialect.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/disc/tools/disc-transform/LinalgExt/LinalgExtDialect.cc.inc"
#include "mlir/disc/tools/disc-transform/LinalgExt/LinalgExtOps.h"

namespace mlir {
namespace disc_ral {
namespace disc_linalg_ext {

void DISCLinalgExtDialect::initialize() {
#define GET_OP_LIST
  addOperations<
#include "mlir/disc/tools/disc-transform/LinalgExt/LinalgExtOps.cc.inc"
      >();
}

/// Materialize an integer or floating point constant.
Operation* DISCLinalgExtDialect::materializeConstant(OpBuilder& builder,
                                                     Attribute value, Type type,
                                                     Location loc) {
  return builder.create<arith::ConstantOp>(loc, value, type);
}

}  // namespace disc_linalg_ext
}  // namespace disc_ral
}  // namespace mlir
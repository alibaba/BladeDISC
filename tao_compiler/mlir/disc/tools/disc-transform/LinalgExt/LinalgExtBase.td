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

#ifndef IREE_DIALECT_LINALGEXT_BASE
#define IREE_DIALECT_LINALGEXT_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

def DISCLinalgExt_Dialect : Dialect {
  let name = "disc_linalg_ext";
  let cppNamespace = "::mlir::disc_ral::disc_linalg_ext";
  let description = [{
    The `disc_linalg_ext` dialect is intended to experiment more support for
    non-structured operations, ie, can not be represented in Linalg operations.
  }];
  let hasCanonicalizer = 1;
  let hasConstantMaterializer = 1;
  let useDefaultAttributePrinterParser = 0;
  let dependentDialects = ["linalg::LinalgDialect"];
}

//===----------------------------------------------------------------------===//
// Type definitions
//===----------------------------------------------------------------------===//

class RankedTensorOrMemRefOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes,
      Or<[IsMemRefTypePred, And<[IsTensorTypePred, HasRankPred]>]>,
  "ranked tensor or memref", "::mlir::ShapedType">;

def AnyRankedTensorOrMemRefType : RankedTensorOrMemRefOf<[AnyType]>;

#endif // IREE_DIALECT_LINALGEXT_BASE

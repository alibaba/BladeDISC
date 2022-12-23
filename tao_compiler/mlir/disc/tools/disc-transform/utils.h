/* Copyright 2022 The BladeDISC Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef DISC_TOOLS_DISC_TRANSFORM_UTILS_H_
#define DISC_TOOLS_DISC_TRANSFORM_UTILS_H_

#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir {
namespace disc_ral {

/// Create a linalg::GenericOp version of an n-D copy that can further tile,
/// lower to loops or vectorize, unlike the current implementation of
/// memref::CopyOp.
Operation* createLinalgCopyOp(OpBuilder& b, Location loc, Value from, Value to,
                              ArrayRef<NamedAttribute> attributes = {});

/// Load transform dialect IR from the given file.
LogicalResult parseTransformModuleFromFile(
    MLIRContext* context, llvm::StringRef transformFileName,
    OwningOpRef<ModuleOp>& transformModule);

// Appends transform dependent dialects.
void addTransformDialectDependentDialects(DialectRegistry& registry);

}  // namespace disc_ral
}  // namespace mlir

#endif  // DISC_TOOLS_DISC_TRANSFORM_UTILS_H_

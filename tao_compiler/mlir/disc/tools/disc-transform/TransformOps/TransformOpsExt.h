// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef DISC_TOOLS_DISC_TRANSFORM_TRANSFORM_OPS_TRANSFORM_OPS_EXT_
#define DISC_TOOLS_DISC_TRANSFORM_TRANSFORM_OPS_TRANSFORM_OPS_EXT_

#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"

namespace mlir {
class DialectRegistry;

namespace func {
class FuncOp;
}  // namespace func

namespace scf {
class ForeachThreadOp;
}  // namespace scf

namespace disc_ral {
/// Registers common transformations that require DISC-specific information
/// into the transform dialect.
void registerTransformDialectCommonExtension(DialectRegistry& registry);

namespace transform_dialect {
// Hook to register common transformations to the transform dialect.
class CommonExtensions
    : public transform::TransformDialectExtension<CommonExtensions> {
 public:
  CommonExtensions();
};
}  // namespace transform_dialect
}  // namespace disc_ral
}  // namespace mlir

#define GET_OP_CLASSES
#include "mlir/disc/tools/disc-transform/TransformOps/TransformOpsExt.h.inc"

#endif  // DISC_TOOLS_DISC_TRANSFORM_TRANSFORM_OPS_TRANSFORM_OPS_EXT_

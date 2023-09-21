// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef DISC_TOOLS_DISC_TRANSFORM_TRANSFORM_OPS_TRANSFORM_OPS_EXT_
#define DISC_TOOLS_DISC_TRANSFORM_TRANSFORM_OPS_TRANSFORM_OPS_EXT_

#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"

namespace mlir {
class DialectRegistry;

namespace func {
class FuncOp;
}  // namespace func

namespace scf {
class ForallOp;
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

/// Pipeline shared memory copy by apply software pipelining scheduling where
/// copy to shared memory is in stage 0 and the rest of the operations are in
/// stage `depth - 1`.
enum class PipeliningSchedulingStrategy {
  // Schedule the load from global memory into stage 0 and the associated store
  // will be in stage depth - 1.
  loadGlobalStage0 = 0,
  // Schedule both the load from global and the store to shared memory in stage
  // 0. The compute operations will be in stage depth-1. This means there won't
  // be vector registers carried between stages.
  loadStoreStage0 = 1,
  // Schedule optimized when using nvidia tensorcore with async copies. It will
  // set all the copies in stage 0 then it will prefecth part of loads in `depth
  // - 2` stage and keep the rest of the load and compute into `depth - 1`.
  nvidiaTensorCore = 2,
};

/// Pipeline copy to shared memory for matmul op
FailureOr<scf::ForOp> applyPipelining(scf::ForOp forOp, int64_t depth,
                                      bool epiloguePeeling,
                                      PipeliningSchedulingStrategy schedule);

LogicalResult optimizeSharedMemoryReadsAndWrites(Operation* parentOp,
                                                 Value memrefValue);

}  // namespace transform_dialect
}  // namespace disc_ral
}  // namespace mlir

#define GET_OP_CLASSES
#include "mlir/disc/tools/disc-transform/TransformOps/TransformOpsExt.h.inc"

#endif  // DISC_TOOLS_DISC_TRANSFORM_TRANSFORM_OPS_TRANSFORM_OPS_EXT_

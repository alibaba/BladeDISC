//===- ParallelLoopTiling.cpp - Tiles scf.parallel ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements loop tiling on parallel loops.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "llvm/ADT/Sequence.h"
#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "tensorflow/compiler/mlir/disc/transforms/codegen_utils.h"

namespace mlir {
namespace disc_ral {

namespace {

using scf::IfOp;
using scf::ParallelOp;

struct ParallelLoopTiling
    : public SCFParallelLoopTilingBase<ParallelLoopTiling> {
  ParallelLoopTiling() = default;
  explicit ParallelLoopTiling(ArrayRef<int64_t> tileSizes,
                              bool withInboundCheck = false) {
    this->tileSizes = tileSizes;
    this->withInboundCheck = withInboundCheck;
  }

  void runOnFunction() override {
    SmallVector<ParallelOp, 2> innermostPloops;
    getInnermostParallelLoops(getFunction().getOperation(), innermostPloops);
    for (ParallelOp ploop : innermostPloops) {
      // FIXME: Add reduction support.
      SmallVector<long> localTileSizes(tileSizes.begin(), tileSizes.end());
      if (ploop.getNumReductions() == 0) {
        lmhlo::FusionOp fusion = ploop->getParentOfType<lmhlo::FusionOp>();
        // TODO: Change this to a assert check after lhlo_fusion pass
        // put even single nodes into a lmhlo.FusionOp
        if (fusion) {
          if (auto attr =
                  fusion->getAttrOfType<IntegerAttr>(kThreadPerBlockHint)) {
            localTileSizes = {attr.getInt()};
          }
        }
        // TODO(zk): we should wrap the tileParallelLoop() from llvm
        // repo after it's merged. This pass should be a wrapper of this
        // function.
        tileParallelLoop(ploop, localTileSizes, withInboundCheck);
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::FunctionPass> createParallelLoopTilingPass(
    ArrayRef<int64_t> tileSizes, bool withInboundCheck) {
  return std::make_unique<ParallelLoopTiling>(tileSizes, withInboundCheck);
}

}  // namespace disc_ral
}  // namespace mlir

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
#include "lhlo/IR/lhlo_ops.h"
#include "llvm/ADT/Sequence.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/codegen_utils.h"
#include "mlir/disc/transforms/fusion_utils.h"
#include "tensorflow/core/util/env_var.h"

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

  void runOnOperation() override {
    SmallVector<ParallelOp, 2> innermostPloops;
    getInnermostParallelLoops(getOperation(), innermostPloops);
    for (ParallelOp ploop : innermostPloops) {
      // FIXME: Add reduction support.
      SmallVector<long> localTileSizes(tileSizes.begin(), tileSizes.end());
      if (ploop.getNumReductions() == 0) {
        lmhlo::FusionOp fusion = ploop->getParentOfType<lmhlo::FusionOp>();
        // TODO: Change this to a assert check after lhlo_fusion pass
        // put even single nodes into a lmhlo.FusionOp
        if (fusion) {
          if (isMemIntensiveOptExperimentalEnabled()) {
            // Do not deal with kStitch fusion.
            auto fusionTypeAttr =
                fusion->getAttrOfType<StringAttr>(kDiscFusionTypeAttrName);
            if (fusionTypeAttr && fusionTypeAttr.getValue() == "kStitch") {
              continue;
            }
          }
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

std::unique_ptr<OperationPass<func::FuncOp>> createParallelLoopTilingPass(
    ArrayRef<int64_t> tileSizes, bool withInboundCheck) {
  return std::make_unique<ParallelLoopTiling>(tileSizes, withInboundCheck);
}

}  // namespace disc_ral
}  // namespace mlir

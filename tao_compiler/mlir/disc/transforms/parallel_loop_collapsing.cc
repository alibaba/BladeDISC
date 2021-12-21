//===- ParallelLoopCollapsing.cpp - Pass collapsing parallel loop indices -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that collapses ParallelOp into 1-D.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/LoopUtils.h"

namespace mlir {
namespace disc_ral {

namespace {
struct ParallelLoopCollapsing
    : public ParallelLoopCollapsingBase<ParallelLoopCollapsing> {
  void runOnFunction() override {
    SmallVector<scf::ParallelOp, 2> innermostPloops;
    getInnermostParallelLoops(getFunction().getOperation(), innermostPloops);
    for (scf::ParallelOp ploop : innermostPloops) {
      if (ploop.getInductionVars().size() > 1) {
        std::vector<unsigned> inds(ploop.getInductionVars().size());
        for (unsigned int id = 0; id < ploop.getInductionVars().size(); id++) {
          inds[id] = id;
        }
        llvm::SmallVector<std::vector<unsigned>, 3> combinedLoops;
        combinedLoops.push_back(inds);
        collapseParallelLoops(ploop, combinedLoops);
      }
    }
  }
};
}  // namespace

std::unique_ptr<mlir::FunctionPass> createDiscParallelLoopCollapsingPass() {
  return std::make_unique<ParallelLoopCollapsing>();
}

}  // namespace disc_ral
}  // namespace mlir

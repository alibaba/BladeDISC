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
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Affine/LoopUtils.h"

namespace mlir {
namespace disc_ral {

namespace {
struct ParallelLoopCollapsing
    : public ParallelLoopCollapsingBase<ParallelLoopCollapsing> {
  void runOnOperation() override {
    SmallVector<scf::ParallelOp, 2> innermostPloops;
    getInnermostParallelLoops(getOperation(), innermostPloops);
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

std::unique_ptr<OperationPass<FuncOp>> createDiscParallelLoopCollapsingPass() {
  return std::make_unique<ParallelLoopCollapsing>();
}

}  // namespace disc_ral
}  // namespace mlir

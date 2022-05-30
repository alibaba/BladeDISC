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

#include "PassDetail.h"
#include "llvm/ADT/Sequence.h"
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/codegen_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/fusion_utils.h"
#include "tensorflow/core/util/env_var.h"

// This pass unrolls and interleaves the inner-most SCF for-loop ops in kStitch
// fusion on GPU.

namespace mlir {
namespace disc_ral {

namespace {

bool getInnermostForLoops(Operation* rootOp,
                          SmallVectorImpl<scf::ForOp>& result) {
  assert(rootOp != nullptr && "Root operation must not be a nullptr.");
  bool rootEnclosesFloops = false;
  for (Region& region : rootOp->getRegions()) {
    for (Block& block : region.getBlocks()) {
      for (Operation& op : block) {
        bool enclosesFloops = getInnermostForLoops(&op, result);
        rootEnclosesFloops |= enclosesFloops;
        if (auto floop = dyn_cast<scf::ForOp>(op)) {
          rootEnclosesFloops = true;

          // Collect For loop if it is an innermost one.
          if (!enclosesFloops) result.push_back(floop);
        }
      }
    }
  }
  return rootEnclosesFloops;
}

struct ForLoopUnrollInterleave
    : public ForLoopUnrollInterleaveBase<ForLoopUnrollInterleave> {
 public:
  void runOnOperation() override {
    func::FuncOp func = getOperation();

    SmallVector<scf::ForOp> target_ops;
    func.walk([&](lmhlo::FusionOp fusion) {
      auto op = fusion.getOperation();
      // Currently, it only unrolls and interleaves loops for kStitch fusion on
      // GPU.
      if (isStitchFusion(op) && isOnGpu(op)) {
        SmallVector<scf::ParallelOp, 2> innermostPloops;
        getInnermostParallelLoops(op, innermostPloops);
        for (auto ploop : innermostPloops) {
          SmallVector<scf::ForOp> fors;
          getInnermostForLoops(ploop.getOperation(), fors);
          target_ops.insert(target_ops.end(), fors.begin(), fors.end());
        }
      }
    });

    for (auto op : target_ops) {
      disc_ral::loopUnrollByFactorAndTryInterleave(op, 4);
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createForLoopUnrollInterleavePass() {
  return std::make_unique<ForLoopUnrollInterleave>();
}

}  // namespace disc_ral
}  // namespace mlir
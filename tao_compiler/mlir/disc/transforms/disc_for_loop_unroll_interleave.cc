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

void assumeAlignment(scf::ForOp op, int tile_size) {
  // Set alignment of buffers accessed by `load` op.
  DenseSet<Value> buffers_to_align;
  op.walk(
      [&](memref::LoadOp load) { buffers_to_align.insert(load.getMemRef()); });
  OpBuilder builder(op);
  for (auto buffer : buffers_to_align) {
    auto definingOp = buffer.getDefiningOp();
    // Note that we are dealing with the innermost for op.
    if (definingOp->getParentOfType<scf::ForOp>() == op) {
      builder.setInsertionPointAfter(definingOp);
    } else {
      builder.setInsertionPointToStart(op.getBody());
    }
    createAlignMemrefWithTile(builder, buffer, tile_size);
  }
}

// This pass unrolls the for loop in kStitch fusion on GPU with the factor of 4.
// The unrolled instructions will be interleaved if possible. Following is an
// example.
//
// Original loop:
// %result = scf.for %arg0 = %0 to %1 step %c256 iter_args(%arg1 = %cst)
//     -> (f32) {
//   %3 = memref.load %2[%arg0] : memref<?xf32, "gpu">
//   %4 = arith.addf %arg1, %3: f32
//   scf.yield %4 : f32
// }
//
// Optimized loops:
// %result0 = scf.for %arg0 = %0 to %peeling step %c256 iter_args(%arg1 = %cst)
//     -> (f32) {
//   %3 = arith.addi %arg0, %c256
//   %4 = arith.addi %arg0, %c512
//   %5 = arith.addi %arg0, %c768
//   %6 = memref.load %2[%arg0] : memref<?xf32, "gpu">
//   %7 = memref.load %2[%3] : memref<?xf32, "gpu">
//   %8 = memref.load %2[%4] : memref<?xf32, "gpu">
//   %9 = memref.load %2[%5] : memref<?xf32, "gpu">
//   %10 = arith.addf %arg1, %6: f32
//   %11 = arith.addf %10, %7: f32
//   %12 = arith.addf %11, %8: f32
//   %13 = arith.addf %12, %9: f32
//   scf.yield %13 : f32
// }
// %result = scf.for %arg0 = %peeling to %1 step %c256
//     iter_args(%arg1 = %result0) -> (f32) {
//   %3 = memref.load %2[%arg0] : memref<?xf32, "gpu">
//   %4 = arith.addf %arg1, %3: f32
//   scf.yield %4 : f32
// }
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
      // TODO: support more types of fusions.
      if (!isOnGpu(op)) {
        return WalkResult::advance();
      }
      if (isFusionType<FusionType::kStitch, FusionType::kLoop>(op)) {
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
      // Assume alignment for vectorization.
      auto fusion = op->getParentOfType<lmhlo::FusionOp>();
      int vector_size = getVectorizeOrTileHint(fusion.getOperation());
      if (vector_size > 1) {
        assumeAlignment(op, vector_size);
      }

      int64_t default_unroll_factor = 4;
      disc_ral::loopUnrollByFactorAndTryInterleave(
          op, vector_size > 1 ? vector_size : default_unroll_factor);
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
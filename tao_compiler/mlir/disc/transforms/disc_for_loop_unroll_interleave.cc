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

#include "lhlo/IR/lhlo_ops.h"
#include "llvm/ADT/Sequence.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/codegen_utils.h"
#include "mlir/disc/transforms/fusion_utils.h"
#include "mlir/disc/transforms/placement_utils.h"
#include "tensorflow/core/util/env_var.h"

// This pass unrolls and interleaves the inner-most SCF for-loop ops in kStitch
// and kLoop fusion on GPU.

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

// Assume the alignment of memrefs on GPU used in the given op.
void assumeAlignmentOnGPU(Operation* op, int tile_size) {
  DenseSet<Value> buffers_to_align;
  DenseSet<Operation*> ops;
  Block* body = op->getBlock();
  body->walk([&](Operation* operation) {
    Value buffer;
    if (auto load = dyn_cast<memref::LoadOp>(operation)) {
      buffer = load.getMemRef();
    } else if (auto store = dyn_cast<memref::StoreOp>(operation)) {
      buffer = store.getMemRef();
    }
    if (buffer && placement_utils::isGpuMemRef(buffer)) {
      buffers_to_align.insert(buffer);
    }
    ops.insert(operation);
  });
  OpBuilder builder(op);
  for (auto buffer : buffers_to_align) {
    auto definingOp = buffer.getDefiningOp();
    if (ops.contains(definingOp)) {
      builder.setInsertionPointAfter(definingOp);
    } else {
      // The `definingOp` is before this block.
      builder.setInsertionPointToStart(body);
    }
    createAlignMemrefWithTile(builder, buffer, tile_size);
  }
}

// This pass unrolls the for loop in kStitch and kLoop fusion on GPU with the
// default factor (which is 4) or given vectorization factor. The unrolled
// instructions will be interleaved if possible. Following is an example.
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
      // Currently, it only unrolls and interleaves loops for kStitch and
      // kLoop fusion on GPU.
      // TODO: support more types of fusions.
      if (!isOnGpu(op)) return;
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
        assumeAlignmentOnGPU(op.getOperation(), vector_size);
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

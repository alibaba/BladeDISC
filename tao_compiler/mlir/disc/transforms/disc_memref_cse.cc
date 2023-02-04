/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// This file implements CSE of memref.load specific for DISC
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/disc/transforms/PassDetail.h"

using mlir::memref::LoadOp;

namespace mlir {
namespace disc_ral {

namespace {

constexpr unsigned c_MAX_ITERATION = 4096;

// This pass performs regular CSE to memref.load ops. This is an optimization
// pass works after LhloLegalizeRootsToParallelLoops & InputInlineFusion.
//
// In general CSE for memref.load should consider a lot more aspects. However,
// this is much simpler in the background of DISC.
class DiscMemRefCSEPass : public DiscMemRefCSEPassBase<DiscMemRefCSEPass> {
 public:
  void runOnOperation() override {
    bool changed = true;
    unsigned iter = 0;
    while (changed && iter++ < c_MAX_ITERATION) {
      runCleanUp();
      changed = runMemRefLoadCSE();
    }
    runCleanUp();
  }

 private:
  bool tryMemRefLoadCSE(LoadOp load);
  bool runMemRefLoadCSE();
  void runCleanUp();
  llvm::DenseSet<Operation*> load_set_;
};

// Replace the use of B with A, if memref.load A and memref.load B
// meet the condition:
//   1) they operate on the same memref & indices;
//   2) A dominant B
bool DiscMemRefCSEPass::tryMemRefLoadCSE(LoadOp load) {
  Block* parent_block = load->getBlock();
  SmallVector<LoadOp, 4> to_be_erased;
  parent_block->walk([&](LoadOp other_load) {
    if ((other_load != load) && (other_load.getMemRef() == load.getMemRef()) &&
        (other_load.getIndices() == load.getIndices())) {
      Operation* ancestor = parent_block->findAncestorOpInBlock(*other_load);
      assert(ancestor != nullptr);
      if (load->isBeforeInBlock(ancestor)) {
        to_be_erased.push_back(other_load);
      }
    }
  });
  if (to_be_erased.size() == 0) {
    load_set_.erase(load);
    return false;
  }
  for (LoadOp other : to_be_erased) {
    load_set_.erase(other);
    other->replaceAllUsesWith(load);
    other->erase();
  }
  load_set_.erase(load);
  return true;
}

bool DiscMemRefCSEPass::runMemRefLoadCSE() {
  func::FuncOp func = getOperation();
  func.walk([&](LoadOp load) { load_set_.insert(load); });
  bool changed = false;
  while (load_set_.size() > 0) {
    LoadOp load = cast<LoadOp>(*load_set_.begin());
    changed |= tryMemRefLoadCSE(load);
  }
  return changed;
}

void DiscMemRefCSEPass::runCleanUp() {
  func::FuncOp func = getOperation();
  OpPassManager cleanupPipeline(OpPassManager("func"));
  cleanupPipeline.addPass(createCSEPass());
  (void)runPipeline(cleanupPipeline, func);
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createDiscMemRefCSEPass() {
  return std::make_unique<DiscMemRefCSEPass>();
}

}  // namespace disc_ral
}  // namespace mlir

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

// This file implements GVN optimization on memref dialect. Currently, it only
// implements partial of CPU memref scalar load/store optimization.

#include "llvm/Support/Debug.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "placement_utils.h"
#include "tensorflow/compiler/mlir/disc/disc_util.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"

namespace mlir {
namespace disc_ral {

namespace {

struct LoadScalarSimplifier : public OpRewritePattern<memref::LoadOp> {
 public:
  LoadScalarSimplifier(MLIRContext* context, DominanceInfo* dominance_info)
      : OpRewritePattern<memref::LoadOp>::OpRewritePattern(context),
        dominance_info_(dominance_info) {}

  LogicalResult matchAndRewrite(memref::LoadOp op,
                                PatternRewriter& rewriter) const override;

 private:
  bool extractScalarIndex(ValueRange indices, int64_t& index) const;
  DominanceInfo* dominance_info_;
};

bool LoadScalarSimplifier::extractScalarIndex(ValueRange indices,
                                              int64_t& index) const {
  if (indices.size() != 1) {
    return false;
  }
  auto op = indices[0].getDefiningOp();
  if (auto constOp = dyn_cast_or_null<arith::ConstantIndexOp>(op)) {
    index = constOp.value();
    return true;
  }
  return false;
}

// Replace the pattern of:
//   memref.store %value, %memref_x[%index_x] : memref<?xtype>
//   %a = memref.load %memref_x[%index_x] : memref<?xtype>
//   Use of `%a`
// with:
//   memref.store %value, %memref_x[%index_x] : memref<?xtype>
//   Use of `%value`
// The store op could be eliminated in other passes (e.g., DCE).
// Note: we only deal with memrefs on CPU in this pass currently.
LogicalResult LoadScalarSimplifier::matchAndRewrite(
    memref::LoadOp op, PatternRewriter& rewriter) const {
  Location loc = op.getLoc();
  int64_t load_index;
  if (!extractScalarIndex(op.getIndices(), load_index)) {
    return failure();
  }

  Value loadMemRef = op.getMemRef();
  if (placement_utils::isGpuMemRef(loadMemRef)) {
    return failure();
  }
  DenseSet<Operation*> writable_ops;
  for (Operation* user : loadMemRef.getUsers()) {
    if (!dominance_info_->dominates(user, op)) {
      continue;
    }
    if (!isa<memref::StoreOp>(user)) {
      if (IsOpWriteValue(user, loadMemRef)) {
        writable_ops.insert(user);
      }
      continue;
    }
    // Load and store should work on the same index.
    auto store = dyn_cast<memref::StoreOp>(user);
    int64_t store_index;
    if (!extractScalarIndex(store.getIndices(), store_index)) {
      continue;
    }
    if (load_index == store_index) {
      auto value = store.value();
      writable_ops.insert(user);
    }
  }
  SmallVector<Operation*> write_frontier;
  for (auto curr : writable_ops) {
    bool dominant_other = false;
    for (auto other : writable_ops) {
      if (!dominance_info_->dominates(curr, other)) {
        dominant_other = true;
        break;
      }
    }
    if (!dominant_other) {
      write_frontier.push_back(curr);
    }
  }
  // Make sure that there is only one deterministic dominant store.
  if ((write_frontier.size() == 1)) {
    auto store = dyn_cast_or_null<memref::StoreOp>(write_frontier[0]);
    if (store != nullptr) {
      rewriter.replaceOp(op, store.value());
      return success();
    }
  }
  return failure();
}

class DiscMemRefGVNPass : public DiscMemRefGVNPassBase<DiscMemRefGVNPass> {
 public:
  void runOnFunction() override {
    FuncOp func = getFunction();
    MLIRContext* ctx = func.getContext();

    // Populate patterns.
    RewritePatternSet patterns(ctx);
    DominanceInfo dominance_info(func);
    patterns.insert<LoadScalarSimplifier>(ctx, &dominance_info);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<FunctionPass> createDiscMemRefGVNPass() {
  return std::make_unique<DiscMemRefGVNPass>();
}

}  // namespace disc_ral
}  // namespace mlir
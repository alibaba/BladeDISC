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

// This file implements load/store optimization on memref dialect. It helps to
// eliminate unnecessary load/stores.

#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "placement_utils.h"

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
// Note: we only deal with memrefs on CPU in this pass currently. Meanwhile,
// we only deal with the case that the loaded memref are only stored once at
// the same index.
LogicalResult LoadScalarSimplifier::matchAndRewrite(
    memref::LoadOp op, PatternRewriter& rewriter) const {
  Location loc = op.getLoc();
  int64_t load_index;
  if (!extractScalarIndex(op.getIndices(), load_index)) {
    return failure();
  }

  Value load_memref = op.getMemRef();
  if (placement_utils::isGpuMemRef(load_memref)) {
    return failure();
  }

  memref::StoreOp store_op;
  for (Operation* user : load_memref.getUsers()) {
    // Before check alias, we first check dominance from `op` to `user`, which
    // helps to reduce the cases of failure caused by alias dominated by `op`.
    if (dominance_info_->dominates(op.getOperation(), user)) {
      continue;
    }
    if (IsMemRefAliasOp(user) ||
        (!isa<memref::StoreOp>(user) && IsOpWriteValue(user, load_memref))) {
      // To prevent indirect modification on `load_memref`.
      return failure();
    }
    // Only store-ops dominate `op` will be considered for optimization.
    if (!isa<memref::StoreOp>(user)) {
      continue;
    } else if (!dominance_info_->dominates(user, op.getOperation())) {
      // Deal with case like:
      //   store a
      //   if (%pred) {
      //     store b
      //   }
      //   load c
      auto store = dyn_cast<memref::StoreOp>(user);
      int64_t store_index;
      // If it is unknown that whether b and c are the same position, return
      // failure.
      if (!extractScalarIndex(store.getIndices(), store_index)) {
        return failure();
      }
      // If b and c are the same position, return failure.
      if (load_index == store_index) {
        return failure();
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
      if (store_op != nullptr) {
        // There is already a store-op write at the same index.
        return failure();
      }
      store_op = store;
    }
  }
  if (store_op != nullptr) {
    rewriter.replaceOp(op, store_op.getValue());
    return success();
  }
  return failure();
}

class DiscMemRefLoadStoreSimplifierPass
    : public DiscMemRefLoadStoreSimplifierPassBase<
          DiscMemRefLoadStoreSimplifierPass> {
 public:
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext* ctx = func.getContext();

    // Populate patterns.
    RewritePatternSet patterns(ctx);
    DominanceInfo dominance_info(func);
    patterns.insert<LoadScalarSimplifier>(ctx, &dominance_info);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createDiscMemRefLoadStoreSimplifierPass() {
  return std::make_unique<DiscMemRefLoadStoreSimplifierPass>();
}

}  // namespace disc_ral
}  // namespace mlir

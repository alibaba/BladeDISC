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

// This pass eliminates unnecessary InsertValueOp and ExtractValueOp of LLVM
// dialect. It helps to reduce unnecessary parameters of fusion functions.

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Location.h"                            // from @llvm-project
#include "mlir/IR/MLIRContext.h"                         // from @llvm-project
#include "mlir/IR/Operation.h"                           // from @llvm-project
#include "mlir/IR/PatternMatch.h"                        // from @llvm-project
#include "mlir/Pass/Pass.h"                              // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"                      // from @llvm-project
#include "mlir/disc/transforms/PassDetail.h"

namespace mlir {
namespace disc_ral {

namespace {

// If an inserted value is not used anymore, rease the insert operator.
struct LLVMInsertSimplifier : public OpRewritePattern<LLVM::InsertValueOp> {
  explicit LLVMInsertSimplifier(MLIRContext* context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(LLVM::InsertValueOp op,
                                PatternRewriter& rewriter) const override {
    // For safety, it only checks the insertvalue that operates on `struct`
    // type. Meanwhile, it requires that the struct is only consumed by either
    // insertvalue or extractvalue.
    Type containerType = op.getContainer().getType();
    if (!containerType.isa<LLVM::LLVMStructType>()) {
      return failure();
    }

    DenseSet<Value> containers;
    auto insertValueOp = op;
    // Propagate back to find all values representing the container.
    while (insertValueOp) {
      auto container = insertValueOp.getContainer();
      containers.insert(container);
      insertValueOp = container.getDefiningOp<LLVM::InsertValueOp>();
    }
    // Propagate forth to find all values representing the container.
    SmallVector<Value> forthInsertChain;
    forthInsertChain.push_back(op);
    while (!forthInsertChain.empty()) {
      Value insertValueOp = forthInsertChain.pop_back_val();
      containers.insert(insertValueOp);
      for (auto user : insertValueOp.getUsers()) {
        if (auto insertOp = dyn_cast_or_null<LLVM::InsertValueOp>(user)) {
          forthInsertChain.push_back(insertOp);
        }
      }
    }

    auto position = op.getPosition();
    // Check whether there are extractvalue operators on the same `position`.
    for (auto container : containers) {
      for (auto user : container.getUsers()) {
        // Make sure the struct is only consumed by either insertvalue or
        // extractvalue.
        if (!isa<LLVM::InsertValueOp, LLVM::ExtractValueOp>(user)) {
          return failure();
        }
        if (auto extract = dyn_cast_or_null<LLVM::ExtractValueOp>(user)) {
          auto pos = extract.getPosition();
          // For simplification, only check the first position.
          if (position[0] == pos[0]) {
            // The same first position consumerd. Current InsertValueOp cannot
            // be erased.
            return failure();
          }
        }
      }
    }

    // Till now, this InsertValueOp is not necessary. To erase it.
    op.replaceAllUsesWith(op.getContainer());

    return success();
  }
};

struct LLVMInsertValueSimplifierPass
    : public LLVMInsertValueSimplifierPassBase<LLVMInsertValueSimplifierPass> {
  void runOnOperation() override {
    LLVM::LLVMFuncOp func = getOperation();

    MLIRContext* ctx = func.getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<LLVMInsertSimplifier>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      func.emitError("applyPatternsAndFoldGreedily does not converge");
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<LLVM::LLVMFuncOp>>
createLLVMInsertValueSimplifierPass() {
  return std::make_unique<LLVMInsertValueSimplifierPass>();
}

}  // namespace disc_ral
}  // namespace mlir

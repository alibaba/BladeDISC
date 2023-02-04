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

#include <unordered_set>
#include <vector>

#include "llvm/ADT/Hashing.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/disc/transforms/PassDetail.h"

namespace mlir {
namespace disc_ral {

namespace {
struct KeyHash {
  std::size_t operator()(Operation* const& opC) const {
    auto* op = const_cast<Operation*>(opC);
    return OperationEquivalence::computeHash(
        op,
        /*hashOperands=*/OperationEquivalence::directHashValue,
        /*hashResults=*/OperationEquivalence::ignoreHashValue,
        OperationEquivalence::IgnoreLocations);
  }
};

// Many codes are copy from llvm-project/mlir/lib/Transforms/CSE.cpp
struct KeyEqual {
  bool isEquivalentTo(Operation* lhs, Operation* rhs,
                      function_ref<LogicalResult(Value, Value)> mapOperands,
                      function_ref<LogicalResult(Value, Value)> mapResults,
                      OperationEquivalence::Flags flags) const {
    if (lhs == rhs) return true;

    // Compare the operation properties.
    if (lhs->getName() != rhs->getName() ||
        lhs->getAttrDictionary() != rhs->getAttrDictionary() ||
        lhs->getNumRegions() != rhs->getNumRegions() ||
        lhs->getNumSuccessors() != rhs->getNumSuccessors() ||
        lhs->getNumOperands() != rhs->getNumOperands() ||
        lhs->getNumResults() != rhs->getNumResults())
      return false;
    if (!(flags & OperationEquivalence::IgnoreLocations) &&
        lhs->getLoc() != rhs->getLoc())
      return false;

    ValueRange lhsOperands = lhs->getOperands(),
               rhsOperands = rhs->getOperands();
    SmallVector<Value> lhsOperandStorage, rhsOperandStorage;
    if (lhs->hasTrait<mlir::OpTrait::IsCommutative>()) {
      lhsOperandStorage.append(lhsOperands.begin(), lhsOperands.end());
      llvm::sort(lhsOperandStorage, [](Value a, Value b) -> bool {
        return a.getAsOpaquePointer() < b.getAsOpaquePointer();
      });
      lhsOperands = lhsOperandStorage;

      rhsOperandStorage.append(rhsOperands.begin(), rhsOperands.end());
      llvm::sort(rhsOperandStorage, [](Value a, Value b) -> bool {
        return a.getAsOpaquePointer() < b.getAsOpaquePointer();
      });
      rhsOperands = rhsOperandStorage;
    }
    auto checkValueRangeMapping =
        [](ValueRange lhs, ValueRange rhs,
           function_ref<LogicalResult(Value, Value)> mapValues) {
          for (auto operandPair : llvm::zip(lhs, rhs)) {
            Value curArg = std::get<0>(operandPair);
            Value otherArg = std::get<1>(operandPair);
            if (curArg.getType() != otherArg.getType()) return false;
            if (failed(mapValues(curArg, otherArg))) return false;
          }
          return true;
        };
    // Check mapping of operands and results.
    if (!checkValueRangeMapping(lhsOperands, rhsOperands, mapOperands))
      return false;
    if (!checkValueRangeMapping(lhs->getResults(), rhs->getResults(),
                                mapResults))
      return false;

    auto lhsReduceOp = dyn_cast<mhlo::ReduceOp>(lhs);
    auto rhsReduceOp = dyn_cast<mhlo::ReduceOp>(rhs);
    if (lhsReduceOp && rhsReduceOp) {
      auto& lhsBody = lhsReduceOp.getBody();
      auto& rhsBody = rhsReduceOp.getBody();
      if (not(lhsBody.hasOneBlock() && rhsBody.hasOneBlock())) return false;

      auto& lhsBlock = lhsBody.front();
      auto& rhsBlock = rhsBody.front();
      auto& lhsOperations = lhsBlock.getOperations();
      auto& rhsOperations = rhsBlock.getOperations();
      if (lhsOperations.size() != rhsOperations.size()) return false;

      auto ignoreValueEquivalenceIfSameBlock = [&](Value lhs, Value rhs) {
        if (lhs.getParentBlock() != &lhsBlock or
            rhs.getParentBlock() != &rhsBlock) {
          return failure();
        }
        return success();
      };
      for (auto it : llvm::zip(lhsOperations, rhsOperations)) {
        auto& lhsOp = std::get<0>(it);
        auto& rhsOp = std::get<1>(it);
        // We currently do not compare equality of nested regions
        if (lhsOp.getNumRegions() != 0 or rhsOp.getNumRegions() != 0)
          return false;

        auto innerOpMatch = OperationEquivalence::isEquivalentTo(
            &lhsOp, &rhsOp,
            /*mapOperands=*/ignoreValueEquivalenceIfSameBlock,
            /*mapResults=*/OperationEquivalence::ignoreValueEquivalence,
            OperationEquivalence::IgnoreLocations);
        if (not innerOpMatch) return false;
      }
      return true;
    }
    if (lhs->getNumRegions() != 0 or rhs->getNumRegions() != 0) return false;

    return true;
  }

  bool operator()(Operation* const& lhsC, Operation* const& rhsC) const {
    auto* lhs = const_cast<Operation*>(lhsC);
    auto* rhs = const_cast<Operation*>(rhsC);

    if (lhs == nullptr or rhs == nullptr) return false;
    return isEquivalentTo(
        lhs, rhs,
        /*mapOperands=*/OperationEquivalence::exactValueMatch,
        /*mapResults=*/OperationEquivalence::ignoreValueEquivalence,
        OperationEquivalence::IgnoreLocations);
  }
};
}  // namespace

namespace {
// The original CSE provided in LLVM will not work on operations with
// body regions such as mhlo::ReduceOp.
//
// The pass add CSE for mhlo::ReduceOp and creates the place to add
// more other mhlo operations in the future.
struct DiscMhloCSEPass : public DiscMhloCSEPassBase<DiscMhloCSEPass> {
  using ScopedMapTy = std::unordered_set<Operation*, KeyHash, KeyEqual>;

  /// Attempt to eliminate a redundant operation.
  void simplifyOperation(ScopedMapTy& knownValues, Operation* op) {
    // One should use CSE pass for common cases.
    // Don't simplify operations other than mhlo::ReduceOp.
    if (!isa<mhlo::ReduceOp>(op)) return;

    // If the operation is already trivially dead just add it to the erase list.
    if (isOpTriviallyDead(op)) {
      opsToErase.push_back(op);
      return;
    }

    // Look for an existing definition for the operation.
    auto existing = knownValues.find(op);
    if (existing != knownValues.end()) {
      replaceUsesAndDelete(knownValues, op, *existing);
      return;
    }
    // Otherwise, we add this operation to the known values map.
    knownValues.insert(op);
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    ScopedMapTy knownValues;
    // ScopedMapTy::ScopeTy scope(knownValues);
    func.walk([&](Operation* op) { simplifyOperation(knownValues, op); });

    // If no operations were erased, then we mark all analyses as preserved.
    if (opsToErase.empty()) return markAllAnalysesPreserved();

    /// Erase any operations that were marked as dead during simplification.
    for (auto* op : opsToErase) op->erase();
    opsToErase.clear();

    // We currently don't remove region operations, so mark dominance as
    // preserved.
    markAnalysesPreserved<DominanceInfo, PostDominanceInfo>();
  }

 private:
  void replaceUsesAndDelete(ScopedMapTy& knownValues, Operation* op,
                            Operation* existing);

  /// Operations marked as dead and to be erased.
  std::vector<Operation*> opsToErase;
};

void DiscMhloCSEPass::replaceUsesAndDelete(ScopedMapTy& knownValues,
                                           Operation* op, Operation* existing) {
  // If the region has SSA dominance, then we are guaranteed to have not
  // visited any use of the current operation.
  op->replaceAllUsesWith(existing);
  opsToErase.push_back(op);

  // If the existing operation has an unknown location and the current
  // operation doesn't, then set the existing op's location to that of the
  // current op.
  if (existing->getLoc().isa<UnknownLoc>() && !op->getLoc().isa<UnknownLoc>())
    existing->setLoc(op->getLoc());
}
}  // namespace
std::unique_ptr<OperationPass<func::FuncOp>> createDiscMhloCSEPass() {
  return std::make_unique<DiscMhloCSEPass>();
}

}  // namespace disc_ral
}  // namespace mlir

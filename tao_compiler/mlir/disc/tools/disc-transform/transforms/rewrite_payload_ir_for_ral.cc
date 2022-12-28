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

#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/disc/tools/disc-transform/transforms/PassDetail.h"
#include "tensorflow/compiler/mlir/disc/transforms/placement_utils.h"

#define DEBUG_TYPE "disc-rewrite-payload-ir-for-ral"

// This file implements the logic to convert the transformed payload IR to be
// suitable for RAL.

namespace mlir {
namespace disc_ral {
namespace {

using func::FuncOp;
using placement_utils::copyWithMemorySpace;
using scf::ForeachThreadOp;
using scf::ParallelOp;

struct DiscRewritePayloadIRForRALPass
    : public DiscRewritePayloadIRForRALPassBase<
          DiscRewritePayloadIRForRALPass> {
  explicit DiscRewritePayloadIRForRALPass(bool gpuEnabled)
      : DiscRewritePayloadIRForRALPassBase<DiscRewritePayloadIRForRALPass>::
            DiscRewritePayloadIRForRALPassBase() {
    this->gpuEnabled_ = gpuEnabled;
  }
  void runOnOperation() override;

  // replace scf::foreach_thread op with scf::parallel op
  LogicalResult convertForeachThreadToParallelOp();
  LogicalResult funcLevelConvertForeachThreadToParallelOp(FuncOp funcOp);

  // assign placement info for each memref value, e.g. memref<f32> ->
  // memref<f32, "cpu">
  LogicalResult assignPlacement();
  LogicalResult assignPlacementForFuncOp(FuncOp funcOp);
};

LogicalResult
DiscRewritePayloadIRForRALPass::funcLevelConvertForeachThreadToParallelOp(
    FuncOp funcOp) {
  SmallVector<ForeachThreadOp> forOps;
  funcOp.walk([&](ForeachThreadOp op) { forOps.push_back(op); });

  OpBuilder b(funcOp);
  for (ForeachThreadOp foreachThreadOp : forOps) {
    if (foreachThreadOp.getOutputs().size() != 0)
      return foreachThreadOp->emitError()
             << "Not support ForeachThreadOp with outputs a.t.m.\n";

    b.setInsertionPoint(foreachThreadOp);
    Location loc = foreachThreadOp.getLoc();
    int64_t rank = foreachThreadOp.getRank();
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value one = b.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value> lowerBounds(rank, zero);
    SmallVector<Value> upperBounds = foreachThreadOp.getNumThreads();
    SmallVector<Value> steps(rank, one);

    auto parallelOp =
        b.create<ParallelOp>(loc, lowerBounds, upperBounds, steps);
    BlockAndValueMapping mapping;
    for (const auto& z : llvm::zip(foreachThreadOp.getThreadIndices(),
                                   parallelOp.getInductionVars()))
      mapping.map(std::get<0>(z), std::get<1>(z));
    b.setInsertionPointToStart(parallelOp.getBody());
    for (auto& nestedOp : foreachThreadOp.getBody()->without_terminator()) {
      Operation* cloned = b.clone(nestedOp, mapping);
    }
    foreachThreadOp->erase();
  }
  return success();
}

LogicalResult
DiscRewritePayloadIRForRALPass::convertForeachThreadToParallelOp() {
  for (auto funcOp : getOperation().getOps<FuncOp>()) {
    if (failed(funcLevelConvertForeachThreadToParallelOp(funcOp)))
      return failure();
  }
  return success();
}

LogicalResult DiscRewritePayloadIRForRALPass::assignPlacementForFuncOp(
    FuncOp funcOp) {
  auto maybeConvertType = [&](Type ty) -> Type {
    auto memrefTy = ty.dyn_cast<MemRefType>();
    if (!memrefTy || memrefTy.getMemorySpace()) return ty;
    return copyWithMemorySpace(
        memrefTy.getContext(), memrefTy,
        this->gpuEnabled_ ? placement_utils::kGpu : placement_utils::kCpu);
  };

  auto convertValue = [&](Value v) {
    auto newTy = maybeConvertType(v.getType());
    if (newTy != v.getType()) v.setType(newTy);
    return success();
  };

  // update types of results of operations
  if (funcOp
          ->walk([&](Operation* op) {
            for (Value value : llvm::to_vector(op->getResults())) {
              if (failed(convertValue(value))) return WalkResult::interrupt();
            }
            return WalkResult::advance();
          })
          .wasInterrupted()) {
    return failure();
  }

  // update types of block arguments
  if (funcOp
          ->walk([&](Block* block) {
            for (Value value : llvm::to_vector((block->getArguments()))) {
              if (failed(convertValue(value))) return WalkResult::interrupt();
            }
            return WalkResult::advance();
          })
          .wasInterrupted()) {
    return failure();
  }

  // update the type of func op
  SmallVector<Type, 4> refinedInputTypes;
  for (Type ty : funcOp.getArgumentTypes()) {
    refinedInputTypes.push_back(maybeConvertType(ty));
  }
  SmallVector<Type, 4> refinedOutputTypes;
  for (Type ty : funcOp.getResultTypes()) {
    refinedOutputTypes.push_back(maybeConvertType(ty));
  }
  auto newFuncTy = FunctionType::get(funcOp.getContext(), refinedInputTypes,
                                     refinedOutputTypes);
  funcOp.setType(newFuncTy);
  return success();
}

LogicalResult DiscRewritePayloadIRForRALPass::assignPlacement() {
  if (gpuEnabled_)
    return getOperation()->emitError()
           << "not support assign placement info for gpu a.t.m.\n";

  for (FuncOp funcOp :
       llvm::to_vector<4>(getOperation().getOps<func::FuncOp>())) {
    if (failed(assignPlacementForFuncOp(funcOp))) return failure();
  }

  return success();
}

void DiscRewritePayloadIRForRALPass::runOnOperation() {
  // 1, rewrite scf.foreach_thread to scf.parallel
  if (failed(convertForeachThreadToParallelOp())) {
    return signalPassFailure();
  }
  LLVM_DEBUG(llvm::dbgs() << "After ForeachThreadOp -> ParallelOp:\n"
                          << getOperation() << "\n");

  // 2, assign placement info for each memref value.
  if (failed(assignPlacement())) {
    return signalPassFailure();
  }
  LLVM_DEBUG(llvm::dbgs() << "After assign placement:\n"
                          << getOperation() << "\n");
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createDiscRewritePayloadIRForRALPass(
    bool gpuEnabled) {
  return std::make_unique<DiscRewritePayloadIRForRALPass>(gpuEnabled);
}

}  // namespace disc_ral
}  // namespace mlir

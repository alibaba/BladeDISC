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
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/disc/tools/disc-transform/transforms/PassDetail.h"
#include "mlir/disc/transforms/placement_utils.h"

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

  // assign placement info for each memref value, e.g. memref<f32> ->
  // memref<f32, "cpu">
  LogicalResult assignPlacement();
  LogicalResult assignPlacementForFuncOp(FuncOp funcOp);
};

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
  // assign placement info for each memref value.
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

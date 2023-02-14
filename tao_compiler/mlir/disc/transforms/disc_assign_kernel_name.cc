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

#include "lhlo/IR/lhlo_ops.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/fusion_utils.h"

namespace mlir {
namespace disc_ral {

using gpu::GPUFuncOp;
using gpu::GPUModuleOp;
using gpu::LaunchFuncOp;
using lmhlo::FusionOp;

namespace {

struct AssignKernelNamePass
    : public AssignKernelNamePassBase<AssignKernelNamePass> {
  void runOnOperation() override {
    SmallVector<FusionOp, 4> ops;
    getOperation().walk([&](FusionOp op) { ops.push_back(op); });

    for (FusionOp op : ops) {
      if (failed(processFusionOp(op))) {
        signalPassFailure();
        return;
      }
    }

    // TODO(disc): this part of code could be removed once we put all data lmhlo
    // ops inside fusion op.
    SmallVector<LaunchFuncOp, 4> launchOps;
    getOperation().walk([&](LaunchFuncOp op) {
      if (!op->getParentOfType<FusionOp>()) launchOps.push_back(op);
    });
    for (auto& en : llvm::enumerate(launchOps)) {
      auto name = (llvm::Twine("gKernel_") + llvm::Twine(en.index()) +
                   llvm::Twine("_") + en.value().getKernelName().getValue())
                      .str();
      processLaunchFuncOp(en.value(), name);
    }
  }

  LogicalResult processFusionOp(FusionOp op);
  LogicalResult processLaunchFuncOp(LaunchFuncOp op, StringRef name);
};

LogicalResult AssignKernelNamePass::processFusionOp(FusionOp op) {
  SmallVector<LaunchFuncOp, 4> gpuOps;
  op.getBody()->walk([&](LaunchFuncOp gpuOp) { gpuOps.push_back(gpuOp); });

  auto fusionName = getFusionFullName(op);
  for (auto&& en : llvm::enumerate(gpuOps)) {
    auto kernelName =
        (fusionName +
         (en.index() != 0 ? ("_" + llvm::Twine(en.index())) : llvm::Twine("")))
            .str();
    if (failed(processLaunchFuncOp(en.value(), kernelName))) return failure();
  }
  return success();
}

LogicalResult AssignKernelNamePass::processLaunchFuncOp(LaunchFuncOp op,
                                                        StringRef name) {
  auto m = getOperation().lookupSymbol<GPUModuleOp>(op.getKernelModuleName());
  auto func = m.lookupSymbol<GPUFuncOp>(op.getKernelName());
  func.setName(name);
  OpBuilder b(op);
  auto newOp = b.create<LaunchFuncOp>(
      op.getLoc(), func, op.getGridSizeOperandValues(),
      op.getBlockSizeOperandValues(), op.getDynamicSharedMemorySize(),
      op.getKernelOperands());
  op.erase();
  return success();
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createDiscAssignKernelNamePass() {
  return std::make_unique<AssignKernelNamePass>();
}

}  // namespace disc_ral
}  // namespace mlir

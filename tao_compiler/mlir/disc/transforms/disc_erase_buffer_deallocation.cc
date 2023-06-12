/* Copyright 2023 The BladeDISC Authors. All Rights Reserved.
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

#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/disc/IR/lhlo_disc_ops.h"
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/PassDetail.h"

// This file implements logic to erase dealloc op for GPU func op.

namespace mlir {
namespace disc_ral {

namespace {

struct DiscEraseBufferDeallocationPass
    : public DiscEraseBufferDeallocationPassBase<
          DiscEraseBufferDeallocationPass> {
  void runOnOperation() override {
    auto funcOp = cast<gpu::GPUFuncOp>(getOperation());

    SmallVector<memref::DeallocOp> deallocOps;
    funcOp.walk([&](memref::DeallocOp op) { deallocOps.push_back(op); });

    for (auto op : deallocOps) {
      op->erase();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<gpu::GPUFuncOp>>
createDiscEraseBufferDeallocationPass() {
  return std::make_unique<DiscEraseBufferDeallocationPass>();
}

}  // namespace disc_ral
}  // namespace mlir
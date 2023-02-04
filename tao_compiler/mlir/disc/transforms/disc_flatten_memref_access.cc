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

// This file implements the logic to flattern memref to 1D format.

#include "llvm/Support/Debug.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/disc/IR/disc_shape_ops.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/codegen_utils.h"
#include "mlir/disc/transforms/lhlo_elemental_utils.h"

namespace mlir {
namespace disc_ral {

namespace {

struct DiscFlattenMemrefAccessPass
    : DiscFlattenMemrefAccessPassBase<DiscFlattenMemrefAccessPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Collect target load/store ops before rewrite to avoid modify while
    // traveling.
    SmallVector<memref::LoadOp, 4> loadOps;
    SmallVector<memref::StoreOp, 4> storeOps;
    func.walk([&](memref::LoadOp op) {
      if (op->getParentOfType<scf::ParallelOp>()) loadOps.push_back(op);
    });
    func.walk([&](memref::StoreOp op) {
      if (op->getParentOfType<scf::ParallelOp>()) storeOps.push_back(op);
    });

    for (memref::LoadOp op : loadOps) {
      if (failed(processLoadOp(op))) {
        return signalPassFailure();
      }
    }

    for (memref::StoreOp op : storeOps) {
      if (failed(processStoreOp(op))) {
        return signalPassFailure();
      }
    }
  }

  LogicalResult processLoadOp(memref::LoadOp op);
  LogicalResult processStoreOp(memref::StoreOp op);
};

LogicalResult DiscFlattenMemrefAccessPass::processLoadOp(memref::LoadOp op) {
  OpBuilder b(op);
  Location loc = op.getLoc();
  Value memref = op.getMemRef();
  auto ty = memref.getType().cast<MemRefType>();
  if (ty.getRank() < 1 || !ty.getLayout().isIdentity()) return success();

  SmallVector<Value> dimSizes = disc_ral::getShapeValues(&b, memref);
  Value linear = b.create<disc_shape::LinearizeOp>(loc, b.getIndexType(),
                                                   op.getIndices(), dimSizes);
  Value newValue = disc_ral::createOffsetLoad(b, loc, memref, linear);
  op->getResult(0).replaceAllUsesWith(newValue);
  op->erase();
  return success();
}

LogicalResult DiscFlattenMemrefAccessPass::processStoreOp(memref::StoreOp op) {
  OpBuilder b(op);
  Location loc = op.getLoc();
  Value memref = op.getMemRef();
  auto ty = memref.getType().cast<MemRefType>();
  if (ty.getRank() < 1 || !ty.getLayout().isIdentity()) return success();

  SmallVector<Value> dimSizes = disc_ral::getShapeValues(&b, memref);
  Value linear = b.create<disc_shape::LinearizeOp>(loc, b.getIndexType(),
                                                   op.getIndices(), dimSizes);
  disc_ral::createOffsetStore(b, loc, op.getValue(), memref, linear);
  op->erase();
  return success();
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createDiscFlattenMemrefAccessPass() {
  return std::make_unique<DiscFlattenMemrefAccessPass>();
}

}  // namespace disc_ral
}  // namespace mlir

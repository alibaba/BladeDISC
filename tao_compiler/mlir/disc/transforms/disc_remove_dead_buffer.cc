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

// This file implements the logic to remove some dead buffers.
// A buffer is dead if it's only assessed by following op:
// - dealloc
// - memref.cast/view/subview

// TODO(disc): remove buffers only have memref.store consumers.

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/PassDetail.h"

namespace mlir {
namespace disc_ral {

using memref::AllocOp;

namespace {

bool isKnownSafeConsumer(Operation* op) {
  // Not consider memref.dim/memref.shape_of ops, supposed that these
  // ops are canonicalized before this pass runs.
  return isa<memref::DeallocOp>(op) || IsMemRefAliasOp(op);
}

bool isDeadBuffer(AllocOp op, SmallVectorImpl<Operation*>& consumers) {
  DenseSet<Operation*> opSet;
  SmallVector<Value, 4> worklist{op.getResult()};

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    for (Operation* user : value.getUsers()) {
      if (!opSet.insert(user).second) continue;
      consumers.push_back(user);
      if (auto view = dyn_cast<ViewLikeOpInterface>(user)) {
        worklist.push_back(view->getResult(0));
      }
    }
  }

  return llvm::all_of(consumers, isKnownSafeConsumer);
}

struct RemoveDeadBufferPass
    : public RemoveDeadBufferPassBase<RemoveDeadBufferPass> {
  void runOnOperation() override {
    SmallVector<AllocOp, 4> candidateBuffers;
    getOperation().walk([&](AllocOp op) { candidateBuffers.push_back(op); });
    for (AllocOp op : candidateBuffers) {
      SmallVector<Operation*, 4> users;
      if (isDeadBuffer(op, users)) {
        op->erase();
        for (Operation* user : users) user->erase();
      }
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createDiscRemoveDeadBufferPass() {
  return std::make_unique<RemoveDeadBufferPass>();
}

}  // namespace disc_ral
}  // namespace mlir

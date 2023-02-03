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

// This pass works after the DiscFusion pass, fuse the splat constant into its
// consumers. The splat const will be duplicated if multiple consumers exist.
// This pass can be regarded as a restricted version of the FusionMerger pass
// of XLA, dealing with splat const as producer only.

#include "lhlo/IR/lhlo_ops.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/disc/transforms/PassDetail.h"

namespace mlir {
namespace disc_ral {

using namespace lmhlo;

namespace {

class DiscFuseSplatConstPass
    : public DiscFuseSplatConstPassBase<DiscFuseSplatConstPass> {
  void runOnOperation() override;

 private:
  void processSplatConst(
      lmhlo::ConstantOp constant,
      DenseMap<lmhlo::FusionOp, SmallVector<Operation*, 4>> users);
};

// step 1, find all the unfused splat constants
// step 2, fuse them in to the consumers
// step 3, if not used anymore, erase the original
void DiscFuseSplatConstPass::runOnOperation() {
  func::FuncOp func = getOperation();
  auto* context = &this->getContext();

  SmallVector<lmhlo::ConstantOp, 4> worklist;
  func.walk([&](lmhlo::ConstantOp constant) {
    if (constant.getValue().isSplat() &&
        (constant->getParentOfType<lmhlo::FusionOp>() == nullptr)) {
      worklist.push_back(constant);
    }
  });
  for (auto constant : worklist) {
    Value memref = constant.getOutput();
    DenseMap<lmhlo::FusionOp, SmallVector<Operation*, 4>> users;
    // TODO: Support patterns like const -> memref.cast -> lmhlo.xxxop
    for (Operation* user : memref.getUsers()) {
      if (user == constant.getOperation()) continue;
      // TODO: support if users are non-fused lmhlo op
      if (isa<lmhlo::LmhloOp>(user)) {
        lmhlo::FusionOp fusion = user->getParentOfType<lmhlo::FusionOp>();
        if (fusion != nullptr) {
          users[fusion].emplace_back(user);
        }
      }
    }
    if (users.size() == 0) continue;
    processSplatConst(constant, users);
  }
}

void DiscFuseSplatConstPass::processSplatConst(
    lmhlo::ConstantOp constant,
    DenseMap<lmhlo::FusionOp, SmallVector<Operation*, 4>> users) {
  for (auto user_item : users) {
    auto orig_alloc =
        dyn_cast<memref::AllocOp>(constant.getOutput().getDefiningOp());
    if (orig_alloc == nullptr) continue;
    OpBuilder builder(orig_alloc);
    auto new_alloc = builder.clone(*orig_alloc.getOperation());
    Value memref = new_alloc->getResult(0);
    builder.setInsertionPointToStart(&user_item.first.getRegion().front());
    builder.create<lmhlo::ConstantOp>(constant.getLoc(), constant.getValue(),
                                      memref);
    for (Operation* user : user_item.second) {
      user->replaceUsesOfWith(constant.getOutput(), memref);
    }
  }

  // Erase the original lmhlo.ConstOp if it has no other users
  bool should_erase = true;
  for (Operation* user : constant.getOutput().getUsers()) {
    if (user == constant.getOperation()) continue;
    should_erase = false;
    break;
  }
  if (should_erase) {
    constant.erase();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createDiscFuseSplatConstPass() {
  return std::make_unique<DiscFuseSplatConstPass>();
}

}  // namespace disc_ral
}  // namespace mlir

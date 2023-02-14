/* Copyright 2022 The BladeDISC Authors. All Rights Reserved.

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

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/fusion_utils.h"
#include "mlir/disc/utils/source_emitter.h"

// This file implements the logic to duplicate some lmhlo operations in order
// to enable more opportunities for fusion and reduce memory footprint.

namespace mlir {
namespace disc_ral {

using func::FuncOp;
using namespace lmhlo;

namespace {

// Duplicate the use of scalar and splat constant for several bcasts. It helps
// to reduce the memory traffic of fusions.
struct DuplicateConstant : public OpRewritePattern<lmhlo::ConstantOp> {
 public:
  DuplicateConstant(MLIRContext* context)
      : OpRewritePattern<lmhlo::ConstantOp>::OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(lmhlo::ConstantOp op,
                                PatternRewriter& rewriter) const override;
};

LogicalResult DuplicateConstant::matchAndRewrite(
    lmhlo::ConstantOp op, PatternRewriter& rewriter) const {
  if (!SourceEmitterCUDA::isBroadcastOnScalarOrSplatConstant(op)) {
    return failure();
  }

  Location loc = op.getLoc();
  auto value = op.getValue();
  Value output = op.getOutput();

  SmallVector<Operation*> bcast_ops;
  for (auto user : output.getUsers()) {
    // Simplifies the dominance checking by only identify the order in the same
    // block.
    if (op->isBeforeInBlock(user) &&
        isa<lmhlo::DynamicBroadcastInDimOp>(user)) {
      bcast_ops.push_back(user);
    }
  }

  bool matched = false;
  // for (auto bcast : bcast_ops.drop_front(1))
  for (std::size_t i = 1; i < bcast_ops.size(); i++) {
    auto bcast = bcast_ops[i];
    // Clone the constant.
    OpBuilder builder(op);
    auto orig_alloc = dyn_cast<memref::AllocOp>(output.getDefiningOp());
    if (orig_alloc == nullptr) {
      continue;
    }
    auto new_alloc = builder.clone(*orig_alloc.getOperation());
    Value memref = new_alloc->getResult(0);
    builder.create<lmhlo::ConstantOp>(loc, value, memref);

    bcast->replaceUsesOfWith(output, memref);
    matched = true;
  }

  return matched ? success() : failure();
}

// Duplicate the use of scalar and splat constant followed with bcasts to
// several users. It helps to reduce the memory traffic of fusions.
struct DuplicateConstantWithBcast : public OpRewritePattern<lmhlo::ConstantOp> {
 public:
  DuplicateConstantWithBcast(MLIRContext* context)
      : OpRewritePattern<lmhlo::ConstantOp>::OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(lmhlo::ConstantOp op,
                                PatternRewriter& rewriter) const override;
};

LogicalResult DuplicateConstantWithBcast::matchAndRewrite(
    lmhlo::ConstantOp op, PatternRewriter& rewriter) const {
  if (!SourceEmitterCUDA::isBroadcastOnScalarOrSplatConstant(op)) {
    return failure();
  }

  Location loc = op.getLoc();
  auto value = op.getValue();
  Value output = op.getOutput();

  SmallVector<Operation*> bcast_ops;
  for (auto user : output.getUsers()) {
    // Simplifies the dominance checking by only identify the order in the same
    // block.
    if (op->isBeforeInBlock(user) &&
        isa<lmhlo::DynamicBroadcastInDimOp>(user)) {
      bcast_ops.push_back(user);
    }
  }

  bool matched = false;
  for (auto bcast_op : bcast_ops) {
    auto bcast = dyn_cast<lmhlo::DynamicBroadcastInDimOp>(bcast_op);
    Value bcast_output = bcast.getOutput();
    SmallVector<Operation*> bcast_users;
    for (auto user : bcast_output.getUsers()) {
      if (bcast->isBeforeInBlock(user)) {
        bcast_users.push_back(user);
      }
    }
    // for (auto user : bcast_users.drop_front(1))
    for (std::size_t i = 1; i < bcast_users.size(); i++) {
      auto user = bcast_users[i];
      OpBuilder builder(bcast);
      // Clone the constant.
      auto orig_const_alloc = dyn_cast<memref::AllocOp>(output.getDefiningOp());
      if (orig_const_alloc == nullptr) {
        continue;
      }
      auto new_const_alloc = builder.clone(*orig_const_alloc.getOperation());
      Value new_const_memref = new_const_alloc->getResult(0);
      builder.create<lmhlo::ConstantOp>(loc, value, new_const_memref);
      // Clone the bcast.
      auto orig_bcast_alloc =
          dyn_cast<memref::AllocOp>(bcast_output.getDefiningOp());
      if (orig_bcast_alloc == nullptr) {
        continue;
      }
      auto new_bcast_alloc = builder.clone(*orig_bcast_alloc.getOperation());
      Value new_bcast_memref = new_bcast_alloc->getResult(0);
      builder.create<lmhlo::DynamicBroadcastInDimOp>(
          loc, new_const_memref, bcast.getOutputDimensions(), new_bcast_memref,
          bcast.getBroadcastDimensions());

      user->replaceUsesOfWith(bcast_output, new_bcast_memref);
      matched = true;
    }
  }

  return matched ? success() : failure();
}

struct DiscDuplicateComputationForFusionPass
    : public DiscDuplicateComputationForFusionPassBase<
          DiscDuplicateComputationForFusionPass> {
  using DiscDuplicateComputationForFusionPassBase<
      DiscDuplicateComputationForFusionPass>::
      DiscDuplicateComputationForFusionPassBase;

  explicit DiscDuplicateComputationForFusionPass(
      bool gpu_enabled, const std::string& fusion_strategy)
      : DiscDuplicateComputationForFusionPassBase<
            DiscDuplicateComputationForFusionPass>::
            DiscDuplicateComputationForFusionPassBase() {
    this->gpu_enabled_ = gpu_enabled;
    this->fusion_strategy_ = fusion_strategy;
  }

  void runOnOperation() override {
    FuncOp func = getOperation();

    // skip shape constraint graph
    if (func.getName() == SymbolicDimMgr::getShapeConstraintGraphFunctionName())
      return;

    // Note that we always use base strategy here. Thus the duplicated ops are
    // always supposed to be fused with other ops. It's a conservative strategy.
    // Re-visit this when necessary.
    auto strategy = makeNewPlacementAwareFusionStrategy(gpu_enabled_, "base");
    if (failed(duplicateBroadcastInDimOp(func, *strategy))) {
      signalPassFailure();
      return;
    }

    if (useTransformSchedule()) {
      auto strategy =
          makeNewPlacementAwareFusionStrategy(gpu_enabled_, "transform_based");
      // const weight can be pre-packed. In some cases, multiple dot ops may
      // share the same const op while use different packed layouts for the
      // weight. we duplicate the weight to make sure that each dot general has
      // its own copy, maximizing the opportunities to do weight pre-packing at
      // compile time.
      if (failed(duplicateConstWeightForDotOp(func, *strategy))) {
        signalPassFailure();
        return;
      }
    }

    if (isMemIntensiveOptExperimentalEnabled()) {
      // Populate patterns.
      MLIRContext* ctx = func.getContext();
      RewritePatternSet patterns(ctx);
      patterns.insert<DuplicateConstant, DuplicateConstantWithBcast>(ctx);
      (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
    }
  }

  LogicalResult duplicateBroadcastInDimOp(FuncOp func,
                                          FusionStrategy& strategy);
  LogicalResult duplicateConstWeightForDotOp(FuncOp func,
                                             FusionStrategy& strategy);
};

// Returns true if the value produced by a cheap computation.
// Here "cheap" means duplication of such computation would not hurt
// performance.
bool isCheapComputation(Value value) {
  auto ty = value.getType().dyn_cast<MemRefType>();
  if (!ty) return false;

  return ty.getRank() == 0 || isConstantMemRef(value);
}

// Basic idea is shown as below:
// convert pattern like:
//   %0 = ... : memref<f32> // scalar buffer
//   %1 = ... : memref<2xi32> // target_shape
//   %2 = ... : memref<?x?xf32>
//   "lmhlo.dynamic_broadcast_in_dim"(%0, %1, %2)
//   "lmhlo.abs"(%2, %3)
//   "lmhlo.dot_general"(%3, %4, %5) // non fusible consumer of %2
//   "lmhlo.exponential"(%5, %6)
//   "lmhlo.add"(%2, %6, %7) // another fusible consumer of %2
//   use(%7)
// to:
//   %0 = ... : memref<f32> // scalar buffer
//   %1 = ... : memref<2xi32> // target_shape
//   %2 = ... : memref<?x?xf32>
//   "lmhlo.dynamic_broadcast_in_dim"(%0, %1, %2)
//   "lmhlo.abs"(%2, %3)
//   "lmhlo.dot_general"(%3, %4, %5) // non fusible consumer of %2
//   "lmhlo.exponential"(%5, %6)
//   %new_2 = ... : memref<?x?xf32>
//   "lmhlo.dynamic_broadcast_in_dim"(%0, %1, %new_2) // duplicate bcast op
//   "lmhlo.add"(%new_2, %6, %7) // another fusible consumer of %2
//   use(%7)
//
// Without this transformation, we will have two fusion patterns:
//  pattern #0: `dynamic_broadcast_in_dim` and `abs`
//  pattern #1: `exponential` and `add`
//  note that pattern #0 and #1 can not be further fused due to `dot_general`
//
//  buffer read + writer analysis:
//    pattern #0: read buffer `%0`, writer buffer `%2`, `%3`
//    pattern #1: read buffer `%2`, `%5`, writer buffer `%7`
//
// After transformation, we also have two fusion patterns:
//  pattern #0: `dynamic_broadcast_in_dim` and `abs`
//  pattern #1: `dynamic_broadcast_in_dim`(duplicated), `exponential` and `add`
//
//  but we have smaller memory footprint:
//    pattern #0: read buffer `%0`, writer buffer `%3`
//    pattern #1: read buffer `%0`, `%5`, writer buffer `%7`
//
//  As you can see from the above analysis, we increase one read of `%0`, which
//  is a scalar buffer, and reduce one read and one write of `%2`, which is
//  larger than `%0`.
//
//  Modern NN networks are usually composed of multiple similar layers. Thus the
//  above patterns are very common especailly when we enable shape constraint ir
//  optimization (if enabled, we will do shape prpagation egaerly, and may
//  further enable cross layer CSE, which in turn increases the chance of the
//  occurrence of the above pattern).
LogicalResult DiscDuplicateComputationForFusionPass::duplicateBroadcastInDimOp(
    FuncOp func, FusionStrategy& strategy) {
  SmallVector<Operation*> ops;
  func->walk([&](Operation* op) {
    if (isa<lmhlo::BroadcastOp, lmhlo::BroadcastInDimOp,
            lmhlo::DynamicBroadcastInDimOp>(op)) {
      ops.push_back(op);
    }
  });
  for (Operation* op : ops) {
    Value in = op->getOperand(0);
    Value out = cast<lmhlo::LmhloOp>(op).getResultBuffer();
    auto operandTy = in.getType().dyn_cast<MemRefType>();
    Operation* allocOp = out.getDefiningOp<memref::AllocOp>();
    if (!operandTy || !allocOp) continue;
    if (!isCheapComputation(in)) continue;
    SmallVector<Operation*> fusibleUsers;
    for (Operation* user : out.getUsers()) {
      if (user == op) continue;
      if (strategy.isFusible(user)) fusibleUsers.push_back(user);
    }
    for (size_t i = 1; i < fusibleUsers.size(); ++i) {
      OpBuilder b(fusibleUsers[i]);
      Operation* clonedAllocOp = b.clone(*allocOp);
      Operation* clonedBcastOp = b.clone(*op);
      clonedBcastOp->replaceUsesOfWith(out, clonedAllocOp->getResult(0));
      fusibleUsers[i]->replaceUsesOfWith(out, clonedAllocOp->getResult(0));
    }
  }
  return success();
}

LogicalResult
DiscDuplicateComputationForFusionPass::duplicateConstWeightForDotOp(
    FuncOp func, FusionStrategy& strategy) {
  SmallVector<lmhlo::DotGeneralOp> dotOps;
  func->walk([&](lmhlo::DotGeneralOp op) {
    if (strategy.isFusible(op.getOperation())) dotOps.emplace_back(op);
  });
  for (lmhlo::DotGeneralOp dotOp : dotOps) {
    Value weight = dotOp.getRhs();
    Value rootWeightMemref = getRootMemRef(weight);
    bool hasOtherUsers = false;
    Operation* constOp = nullptr;
    for (Operation* user : getValueUsers(rootWeightMemref)) {
      if (user == dotOp.getOperation()) continue;
      if (isa<lmhlo::ConstantOp>(user)) {
        if (constOp)
          return dotOp->emitError()
                 << "weight buffer consumsed by mulitple const ops\n";
        constOp = user;
        continue;
      }
      hasOtherUsers = true;
    }
    if (!constOp || !hasOtherUsers) continue;
    auto weightTy = weight.getType().cast<MemRefType>();
    auto constTy = constOp->getOperand(0).getType().cast<MemRefType>();
    // TODO(wyzero): support the case where there are cast ops between const op
    // and dot op.
    if (weightTy != constTy) continue;
    OpBuilder b(dotOp);
    Location loc = dotOp->getLoc();
    Value newWeight = b.create<memref::AllocOp>(loc, weightTy);
    Operation* clonedConstOp = b.clone(*constOp);
    clonedConstOp->replaceUsesOfWith(constOp->getOperand(0), newWeight);
    dotOp->replaceUsesOfWith(weight, newWeight);
  }
  return success();
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createDiscDuplicateComputationForFusionPass(
    bool gpu_enabled, const std::string& fusion_strategy) {
  return std::make_unique<DiscDuplicateComputationForFusionPass>(
      gpu_enabled, fusion_strategy);
}

}  // namespace disc_ral
}  // namespace mlir

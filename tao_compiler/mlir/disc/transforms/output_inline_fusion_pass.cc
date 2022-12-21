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

// This file implements output inline fusion pass.
//
#include <limits>

#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/lhlo/transforms/map_lmhlo_to_scalar_op.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/disc/IR/lhlo_disc_ops.h"
#include "tensorflow/compiler/mlir/disc/disc_util.h"
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"
#include "tensorflow/compiler/mlir/disc/transforms/fusion_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/lhlo_elemental_utils.h"

using mlir::memref::LoadOp;

namespace mlir {
namespace disc_ral {

using namespace lmhlo;

namespace {

class OutputInlineFusion
    : public OutputInlineFusionPassBase<OutputInlineFusion> {
  void runOnOperation() override;
};

template <typename LHLO_OpTy>
bool miscFuseHelper(PatternRewriter& rewriter, scf::ParallelOp parallel_op,
                    Operation* consumer, memref::StoreOp store_op,
                    std::vector<memref::StoreOp>& store_ops) {
  // TODO: Implement generic output fusion
  return false;
}

template <>
bool miscFuseHelper<lmhlo::DynamicGatherOp>(
    PatternRewriter& rewriter, scf::ParallelOp parallel_op, Operation* consumer,
    memref::StoreOp store_op, std::vector<memref::StoreOp>& store_ops) {
  if (!isa<lmhlo::DynamicGatherOp>(consumer)) return false;
  llvm::dbgs() << "Enter miscFuseHelper<lmhlo::DynamicGatherOp> \n";
  consumer->dump();
  llvm::dbgs() << "**********************************************\n";
  auto gather_op = dyn_cast<lmhlo::DynamicGatherOp>(consumer);
  auto operand = gather_op.getOperand();
  auto operand_ty = operand.getType().dyn_cast<MemRefType>();
  int64_t operand_rank = operand_ty.getRank();
  auto start_indices = gather_op.getStartIndices();
  auto dimension_numbers = gather_op.getDimensionNumbers();

  auto loc = consumer->getLoc();
  rewriter.setInsertionPoint(store_ops.front().getOperation());

  parallel_op->getParentOfType<func::FuncOp>()->dump();
  Value indice;
  // Here we only support one store_op
  Value v = store_op.getValueToStore();
  Operation* op = v.getDefiningOp();
  if (isa<arith::IndexCastOp>(op)) {
    indice = cast<arith::IndexCastOp>(op).getOperand();
  } else {
    indice = v;
  }

  SmallVector<Value, 4> load_index, store_index;
  if (operand_rank == 1) {
    load_index.push_back(indice);
    store_index.push_back(store_op.getIndices()[0]);
    Value store_value = rewriter.create<memref::LoadOp>(
        loc, gather_op.getOperand(), load_index);
    rewriter.create<memref::StoreOp>(loc, store_value, gather_op.getOutput(),
                                       store_index);
  } else if (operand_rank == 2) {
    for (int i = 0; i < operand_rank; i++) {
      auto rank_index = rewriter.create<arith::ConstantIndexOp>(loc, i);
      load_index.clear();
      load_index.push_back(indice);
      load_index.push_back(rank_index);
      store_index.clear();
      store_index.push_back(store_op.getIndices()[0]);
      store_index.push_back(rank_index);
      Value store_value = rewriter.create<memref::LoadOp>(
          loc, gather_op.getOperand(), load_index);
      rewriter.create<memref::StoreOp>(loc, store_value, gather_op.getOutput(),
                                       store_index);
    }
  }

  llvm::dbgs() << "Exit miscFuseHelper<lmhlo::DynamicGatherOp> \n";
  return true;
}

template <>
bool miscFuseHelper<lmhlo::DynamicReshapeOp>(
    PatternRewriter& rewriter, scf::ParallelOp parallel_op, Operation* consumer,
    memref::StoreOp store_op, std::vector<memref::StoreOp>& store_ops) {
  llvm::dbgs() << "Enter miscFuseHelper<lmhlo::DynamicReshapeOp> \n";
  if (!isa<lmhlo::DynamicReshapeOp>(consumer)) {
    llvm::dbgs() << "Fail miscFuseHelper<lmhlo::DynamicReshapeOp> \n";
    return false;
  }
  llvm::dbgs() << "In miscFuseHelper<lmhlo::DynamicReshapeOp> 0\n";
  auto reshape_op = dyn_cast<lmhlo::DynamicReshapeOp>(consumer);

  auto loc = consumer->getLoc();
  rewriter.setInsertionPoint(store_ops.front().getOperation());
  llvm::dbgs() << "In miscFuseHelper<lmhlo::DynamicReshapeOp> 1\n";

  Value operand_memref = reshape_op->getOperand(0);
  auto operand_shape = getShapeValues(&rewriter, operand_memref);
  Value output_memref = reshape_op->getOperand(2);
  auto output_shape = getShapeValues(&rewriter, output_memref);

  SmallVector<Value, 4> indices;
  for (auto store_op : store_ops) {
    Value linear_index =
        calcLinearIndex(&rewriter, loc, store_op.getIndices(), operand_shape);
    auto output_index =
        calcMultiDimIndex(&rewriter, loc, linear_index, output_shape);
    rewriter.create<memref::StoreOp>(loc, store_op.getValueToStore(),
                                     output_memref, output_index);
  }

  llvm::dbgs() << "Exit miscFuseHelper<lmhlo::DynamicReshapeOp> \n";
  return true;
}

// Currently we do not actually need lower_config, just leave here for potential
// future works.
class OutputInlineFusionPattern : public RewritePattern {
 public:
  explicit OutputInlineFusionPattern(MLIRContext* context,
                                     LowerConfig* lower_config = nullptr)
      : RewritePattern(lmhlo::FusionOp::getOperationName(), /* benefit */ 1,
                       context),
        lower_config_(lower_config) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override {
    // Currently only support cpu kWhere fusions.
    if (!isFusionType<FusionType::kWhere>(op) || isOnGpu(op)) return failure();

    // skip if not the most outter ParallelOp
    auto fusion = cast<lmhlo::FusionOp>(op);
    auto& parent_block = fusion.getRegion().front();

    SmallVector<scf::ParallelOp, 2> innermostPloops;
    getInnermostParallelLoops(op, innermostPloops);

    llvm::dbgs() << "innermostPloops size: " << innermostPloops.size() << "\n";
    // Returns success if any of parallelOp is processed.
    for (scf::ParallelOp parallelOp : innermostPloops) {
      if (!failed(processParallelOp(parallelOp, &parent_block, rewriter))) {
        llvm::dbgs() << "****************success**************** \n";
        auto func = parallelOp->getParentOfType<func::FuncOp>();
        func->dump();
        llvm::dbgs() << "****************success**************** \n";
        return success();
      }
      llvm::dbgs() << "****************fail**************** \n";
      auto func = parallelOp->getParentOfType<func::FuncOp>();
      func->dump();
      llvm::dbgs() << "****************fail**************** \n";
    }
    return failure();
  }

  LogicalResult processParallelOp(scf::ParallelOp parallel_op,
                                  Block* parent_block,
                                  PatternRewriter& rewriter) const {
    llvm::dbgs() << "Enter processParallelOp \n";
    SmallVector<memref::StoreOp, 4> store_ops;
    parallel_op->walk(
        [&](memref::StoreOp store_op) { store_ops.push_back(store_op); });

    llvm::dbgs() << "store_ops size: " << store_ops.size() << "\n";
    for (auto store_op : store_ops) {
      Operation* lhlo_op = getStorableOperation(store_op);
      if (!lhlo_op) continue;

      std::vector<memref::StoreOp> store_ops_in_block =
          collectStoreToSameMemref(store_op);

      llvm::dbgs() << "store_ops_in_block size: " << store_ops_in_block.size()
                   << "\n";

      if (!failed(inlineFuseLhloOp(rewriter, parallel_op, lhlo_op, store_op,
                                   store_ops_in_block))) {
        // TODO: Check whether it is safe to erase this op like input-inline
        rewriter.eraseOp(lhlo_op);
        // lhlo_op->erase();
#if 0
        for (auto store_op : store_ops_in_block) {
          // store_op.erase();
          rewriter.eraseOp(store_op);
        }
#else
        for (auto store_op : store_ops_in_block) {
          auto store_op_users =
              getValueUsersInFusionLike(store_op.getMemRef(), store_op);
          // llvm::dbgs() << "************erase******************** \n";
          // for (auto user : store_op_users) {
          //   user->dump();
          // }
          if (store_op_users.size() == 0) {
            rewriter.eraseOp(store_op);
          }
        }
#endif
        // cleanUnusedLhloOps(parent_block, &rewriter);
        auto func = parallel_op->getParentOfType<func::FuncOp>();
        func->dump();
        return success();
      }
    }

    // Since we did not clean lhlo ops in InputInlineFusionPass for kWhere fusion
    // for possible output fusion, we need to do clean here.
    cleanUnusedLhloOps(parent_block, &rewriter);
    return failure();
  }

 private:
  Operation* getStorableOperation(memref::StoreOp store_op) const {
    llvm::dbgs() << "enter getStorableOperation \n";
    store_op.getOperation()->dump();
    Operation* lhlo_op = nullptr;
    auto users = getValueUsersInFusionLike(store_op.getMemRef(), store_op);
    llvm::dbgs() << "users size: " << users.size() << "\n";
    for (Operation* user : users) {
      if (isa<LmhloOp>(user)) {
        auto user_operand_index = isa<lmhlo::DynamicGatherOp>(user) ? 1 : 0;
        if (isSameUnderlineBuffer(
              cast<LmhloOp>(user).getOperation()->getOperand(user_operand_index),
              store_op.getMemRef())) {
          if (lhlo_op) {
            if (isa<lmhlo::DynamicGatherOp>(user)) {
              return lhlo_op;
            }
            llvm::dbgs() << store_op.getOperation()->getLoc()
                         << " store op find more than one user ("
                         << lhlo_op->getName() << " and " << user->getName()
                         << ")";
            // llvm::report_fatal_error(
            //     "More than one lhlo_op write to one Memref within one fusion");
          }
          lhlo_op = user;
        }
      }
    }
    return lhlo_op;
  }

  // Collect store ops write to the same memref and within the same block as
  // store_op, including store_op itself.
  std::vector<memref::StoreOp> collectStoreToSameMemref(
      memref::StoreOp store_op) const {
    llvm::dbgs() << "Enter collectStoreToSameMemref\n";
    Block* block = store_op.getOperation()->getBlock();
    std::vector<memref::StoreOp> store_ops;
    for (auto& op : block->getOperations()) {
      if (!isa<memref::StoreOp>(&op)) continue;
      llvm::dbgs() << "************************************* \n";
      op.dump();
      llvm::dbgs() << "************************************* \n";

      auto store_op_in_block = dyn_cast<memref::StoreOp>(&op);
      llvm::dbgs() << "************************************* \n";
      store_op.getMemRef().dump();
      store_op_in_block.getMemRef().dump();
      llvm::dbgs() << "************************************* \n";

      if (store_op_in_block.getMemRef() != store_op.getMemRef()) continue;
      store_ops.push_back(store_op_in_block);
    }
    llvm::dbgs() << "Exit collectStoreToSameMemref\n";
    return store_ops;
  }

  LogicalResult inlineFuseLhloOp(
      PatternRewriter& rewriter, scf::ParallelOp parallel_op,
      Operation* consumer, memref::StoreOp store_op,
      std::vector<memref::StoreOp>& store_ops) const {
    if (miscFuseHelper<lmhlo::DynamicReshapeOp>(rewriter, parallel_op, consumer,
                                                store_op, store_ops) ||
        miscFuseHelper<lmhlo::DynamicGatherOp>(rewriter, parallel_op, consumer,
                                               store_op, store_ops)) {
      return success();
    }
    return failure();
  }

  LowerConfig* lower_config_;
};

}  // end anonymous namespace

// This pass works after LhloLegalizeRootsToParallelLoops and InputInlineFusion
// passes
std::unique_ptr<OperationPass<func::FuncOp>>
createDiscOutputInlineFusionPass() {
  return std::make_unique<OutputInlineFusion>();
}

namespace {

constexpr unsigned c_MAX_ITERATION = 4096 * 1000;

void OutputInlineFusion::runOnOperation() {
  func::FuncOp func = getOperation();
  auto* context = &this->getContext();
  RewritePatternSet patterns(context);
  patterns.insert<OutputInlineFusionPattern>(context);

  // Just apply the patterns greedily.
  // There should always be one scf.ParallelOp in the fusion.
  auto config = GreedyRewriteConfig();
  config.maxIterations = c_MAX_ITERATION;
  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns), config))) {
    signalPassFailure();
  }
}

}  // namespace

}  // namespace disc_ral
}  // namespace mlir

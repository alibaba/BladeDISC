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
  auto gather_op = dyn_cast<lmhlo::DynamicGatherOp>(consumer);
  auto operand = gather_op.getOperand();
  auto operand_ty = operand.getType().dyn_cast<MemRefType>();
  int64_t operand_rank = operand_ty.getRank();
  auto start_indices = gather_op.getStartIndices();
  auto dimension_numbers = gather_op.getDimensionNumbers();

  auto loc = consumer->getLoc();
  rewriter.setInsertionPoint(store_ops.front().getOperation());

  // parallel_op->getParentOfType<func::FuncOp>()->dump();
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

  return true;
}

template <>
bool miscFuseHelper<lmhlo::DynamicReshapeOp>(
    PatternRewriter& rewriter, scf::ParallelOp parallel_op, Operation* consumer,
    memref::StoreOp store_op, std::vector<memref::StoreOp>& store_ops) {
  if (!isa<lmhlo::DynamicReshapeOp>(consumer)) {
    return false;
  }
  auto reshape_op = dyn_cast<lmhlo::DynamicReshapeOp>(consumer);

  auto loc = consumer->getLoc();

  llvm::dbgs() << "Dump before DynamicReshapeOp inline\n";
  parallel_op.getOperation()->dump();

  Value operand_memref = reshape_op->getOperand(0);
  Value output_memref = reshape_op->getOperand(2);
  auto output_ty = output_memref.getType().dyn_cast<MemRefType>();
  int64_t output_rank = output_ty.getRank();

  llvm::dbgs() << "store_ops size: " << store_ops.size() << "\n";
  for (auto store : store_ops) {
    store.getOperation()->dump();
    Block* block = store.getOperation()->getBlock();
    SmallVector<memref::LoadOp, 2> load_ops;
    block->walk([&](memref::LoadOp load_op_in_block) {
      if (load_op_in_block.getMemRef() == operand_memref) {
        load_ops.push_back(load_op_in_block);
      }
    });
#if 1
    // We need to know if there exists a += compute(a)
    // We can simply know if so if fusion is a kSparseReduction
    // If so, we need to create a init loop for this memref
    if (load_ops.size() == 1 /*&& fusion_type == FusionType::kSparseReduction*/) {
      llvm::dbgs() << "Create memset output to 0\n";
      rewriter.setInsertionPoint(parallel_op.getOperation());
      Value zero_floating = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getFloatAttr(output_ty.getElementType(), 0));
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      auto output_shape = getShapeValues(&rewriter, output_memref);

      // memset output to 0
      auto num_elements =
          emitNumElementsComputation(rewriter, loc, output_memref);
      parallel_op.getOperation()->dump();
      auto for_op = rewriter.create<scf::ForOp>(loc, /* lowerBound */ zero,
                                                /* upperBound */ num_elements,
                                                /* step */ one);
      for_op.getBody()->clear();
      rewriter.setInsertionPointToStart(for_op.getBody());

      Value i = for_op.getInductionVar();
      auto index = calcMultiDimIndex(&rewriter, loc, i, output_shape);
      rewriter.create<memref::StoreOp>(loc, zero_floating, output_memref, index);
      rewriter.create<scf::YieldOp>(loc, ValueRange({}));
      rewriter.setInsertionPointAfter(for_op);
    }
#endif

    llvm::dbgs() << "load_ops size: " << load_ops.size() << "\n";
    for (auto load : load_ops) {
      llvm::dbgs() << "inline DynamicReshape rewrite load op \n";
      rewriter.setInsertionPointAfter(load.getOperation());
      auto operand_shape = getShapeValues(&rewriter, operand_memref);
      auto output_shape = getShapeValues(&rewriter, output_memref);
      bool same_indices = load.getIndices() == store.getIndices();
      Value linear_index = calcLinearIndex(
          // &rewriter, loc, same_indices ? store.getIndices() :
          // load.getIndices(),
          &rewriter, loc, load.getIndices(), operand_shape);
      SmallVector<Value> output_index =
          calcMultiDimIndex(&rewriter, loc, linear_index, output_shape);
      auto new_load =
          rewriter.create<memref::LoadOp>(loc, output_memref, output_index);
      rewriter.replaceOp(load, new_load.getResult());
    }
    llvm::dbgs() << "inline DynamicReshape computation \n";
    rewriter.setInsertionPointAfter(store.getOperation());
    auto operand_shape = getShapeValues(&rewriter, operand_memref);
    auto output_shape = getShapeValues(&rewriter, output_memref);
    Value linear_index =
        calcLinearIndex(&rewriter, loc, store.getIndices(), operand_shape);
    SmallVector<Value> output_index =
        calcMultiDimIndex(&rewriter, loc, linear_index, output_shape);
#if 0
    llvm::dbgs() << "inline DynamicReshape before create store\n";
    parallel_op.getOperation()->dump();
    parallel_op->getParentOfType<func::FuncOp>()->dump();
    store.getValueToStore().dump();
    for (auto index : output_index) {
      index.dump();
    }
    output_memref.dump();
#endif
    rewriter.create<memref::StoreOp>(loc, store.getValueToStore(),
                                     output_memref, output_index);
    llvm::dbgs() << "inline DynamicReshape computation done for single store\n";
  }
  llvm::dbgs() << "Dump after DynamicReshapeOp inline\n";
  parallel_op.getOperation()->dump();

  return true;
}

template <>
bool miscFuseHelper<lmhlo::DynamicBroadcastInDimOp>(
    PatternRewriter& rewriter, scf::ParallelOp parallel_op, Operation* consumer,
    memref::StoreOp store_op, std::vector<memref::StoreOp>& store_ops) {
  llvm::dbgs() << " DynamicBroadcastInDimOp \n";
  parallel_op.getOperation()->dump();
  llvm::dbgs() << " DynamicBroadcastInDimOp consumer\n";
  consumer->dump();
  llvm::dbgs() << " DynamicBroadcastInDimOp store_op \n";
  store_op.getOperation()->dump();
  llvm::dbgs() << " DynamicBroadcastInDimOp store_ops \n";
  for (auto store : store_ops) {
    store.getOperation()->dump();
  }
  llvm::dbgs() << " DynamicBroadcastInDimOp dump end \n";

  if (!isa<lmhlo::DynamicBroadcastInDimOp>(consumer)) {
    return false;
  }
  auto broadcast_op = dyn_cast<lmhlo::DynamicBroadcastInDimOp>(consumer);

  auto broadcast_dimensions =
      broadcast_op.getBroadcastDimensions().template getValues<int64_t>();
  Value operand_memref = broadcast_op->getOperand(0);
  auto operand_ty = operand_memref.getType().dyn_cast<MemRefType>();
  int64_t input_rank = operand_ty.getRank();
  auto input_shape = operand_ty.getShape();
  Value output_memref = broadcast_op->getOperand(2);
  auto output_ty = output_memref.getType().dyn_cast<MemRefType>();
  int64_t output_rank = output_ty.getRank();
  auto output_shape = output_ty.getShape();

  auto loc = consumer->getLoc();

  SmallVector<Value, 4> indices;
  auto broadcast_dim_size = broadcast_dimensions.size();
  llvm::dbgs() << "output rank: " << output_rank << "\n";
  llvm::dbgs() << "broadcast_dimensions[0]: " << broadcast_dimensions[0] << "\n";
  for (auto store : store_ops) {
    if (broadcast_dim_size == 1) {
      // broadcast from vector
      rewriter.setInsertionPointAfter(store.getOperation());
      auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      for (int dim = 0; dim < output_rank; ++dim) {
        int64_t static_dim_size = output_shape[dim];
        if (static_dim_size == 1) {
          indices.push_back(zero);
        } else {
          if (dim == broadcast_dimensions[0]) {
            indices.push_back(store.getIndices()[0]);
          } else {
            // NOTE: hack here
            // TODO: find a better impl
            indices.push_back(parallel_op.getInductionVars()[1]);
          }
        }
      }
      rewriter.create<memref::StoreOp>(loc, store.getValueToStore(),
                                       output_memref, indices);
    } else {
      // TODO(lanbo.llb): Support more bcast dimensions
      return false;
    }
    indices.clear();
  }
  llvm::dbgs() << "Dump after DynamicBroadcastInDimOp inline\n";
  parallel_op.getOperation()->dump();

  return true;
}

template <>
bool miscFuseHelper<lmhlo::SelectOp>(PatternRewriter& rewriter,
                                     scf::ParallelOp parallel_op,
                                     Operation* consumer,
                                     memref::StoreOp store_op,
                                     std::vector<memref::StoreOp>& store_ops) {
  llvm::dbgs() << " SelectOp \n";
  consumer->dump();
  llvm::dbgs() << " SelectOp store_op \n";
  store_op.getOperation()->dump();
  llvm::dbgs() << " SelectOp store_ops \n";
  for (auto store : store_ops) {
    store.getOperation()->dump();
  }
  llvm::dbgs() << " SelectOp dump end \n";

  if (!isa<lmhlo::SelectOp>(consumer)) {
    return false;
  }

  auto loc = consumer->getLoc();
  auto select_op = dyn_cast<lmhlo::SelectOp>(consumer);
  auto pred_memref = select_op.getOperand(0);
  auto true_memref = select_op.getOperand(1);
  auto false_memref = select_op.getOperand(2);
  auto output_memref = select_op.getOperand(3);
  auto output_ty = output_memref.getType().dyn_cast<MemRefType>();

  Value zero_floating = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getFloatAttr(output_ty.getElementType(), 0));
  // TODO: check whether true is from broadcasted scalar constant
  for (auto store : store_ops) {
    llvm::dbgs() << "Dump store op in SelectOp inline\n";
    store.getOperation()->dump();
    memref::StoreOp false_store_op;
    Block* block = store.getOperation()->getBlock();
    block->walk([&](memref::StoreOp store_op_in_block) {
      if (store_op_in_block.getMemRef() == false_memref) {
        false_store_op = store_op_in_block;
      }
    });
#if 1
    block->walk([&](memref::LoadOp load_op_in_block) {
      if (load_op_in_block.getMemRef() == false_memref) {
        rewriter.setInsertionPointAfter(load_op_in_block.getOperation());
        auto new_load = rewriter.create<memref::LoadOp>(loc, output_memref,
                                                        store.getIndices());
        rewriter.replaceOp(load_op_in_block, new_load.getResult());
#if 1
        // We need to know if there exists a += compute(a)
        // We can simply know if so if fusion is a kSparseReduction
        // If so, we need to create a init loop for this memref
        llvm::dbgs() << "Create memset output to 0\n";
        rewriter.setInsertionPoint(parallel_op.getOperation());
        Value zero_floating = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getFloatAttr(output_ty.getElementType(), 0));
        Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        auto output_shape = getShapeValues(&rewriter, output_memref);

        // memset output to 0
        auto num_elements =
            emitNumElementsComputation(rewriter, loc, output_memref);
        parallel_op.getOperation()->dump();
        auto for_op = rewriter.create<scf::ForOp>(loc, /* lowerBound */ zero,
                                                  /* upperBound */ num_elements,
                                                  /* step */ one);
        for_op.getBody()->clear();
        rewriter.setInsertionPointToStart(for_op.getBody());

        Value i = for_op.getInductionVar();
        auto index = calcMultiDimIndex(&rewriter, loc, i, output_shape);
        rewriter.create<memref::StoreOp>(loc, zero_floating, output_memref, index);
        rewriter.create<scf::YieldOp>(loc, ValueRange({}));
        rewriter.setInsertionPointAfter(for_op);
#endif
      }
    });
#endif
    rewriter.setInsertionPointAfter(false_store_op.getOperation());
    auto select_val = rewriter.create<arith::SelectOp>(
        loc, store.getValueToStore(), zero_floating,
        false_store_op.getValueToStore());
    rewriter.create<memref::StoreOp>(loc, select_val, output_memref,
                                     store.getIndices());
    // We also need to erase memref.store that store
    // to true_memref/false_memref
    rewriter.eraseOp(false_store_op);
  }
  llvm::dbgs() << "Dump after SelectOp inline\n";
  parallel_op.getOperation()->dump();

  return true;
}

bool isSparseFusion(Operation* op) {
  return (isFusionType<FusionType::kWhere>(op) ||
          isFusionType<FusionType::kSparseReduction>(op)) &&
         !isOnGpu(op);
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
    llvm::dbgs() << "Enter matchAndRewrite\n";
    // Currently only support cpu sparse fusions.
    if (!isSparseFusion(op)) return failure();

    // skip if not the most outter ParallelOp
    auto fusion = cast<lmhlo::FusionOp>(op);
    auto& parent_block = fusion.getRegion().front();

    SmallVector<scf::ParallelOp, 2> innermostPloops;
    getInnermostParallelLoops(op, innermostPloops);

    // Returns success if any of parallelOp is processed.
    for (scf::ParallelOp parallelOp : innermostPloops) {
      if (!failed(processParallelOp(parallelOp, &parent_block, rewriter))) {
        // llvm::dbgs() << "****************success**************** \n";
        // parallelOp->getParentOfType<func::FuncOp>()->dump();
        // llvm::dbgs() << "****************success**************** \n";
        return success();
      }
      // llvm::dbgs() << "****************fail**************** \n";
      // parallelOp->getParentOfType<func::FuncOp>()->dump();
      // llvm::dbgs() << "****************fail**************** \n";
    }
    return failure();
  }

  LogicalResult processParallelOp(scf::ParallelOp parallel_op,
                                  Block* parent_block,
                                  PatternRewriter& rewriter) const {
    SmallVector<memref::StoreOp, 4> store_ops;
    parallel_op->walk(
        [&](memref::StoreOp store_op) { store_ops.push_back(store_op); });

    llvm::dbgs() << "Dump store in parallel op\n";
    for (auto store : store_ops) {
      store.getOperation()->dump();
    }
    llvm::dbgs() << "Dump store in parallel op end\n";

    for (auto store_op : store_ops) {
      Operation* lhlo_op = getStorableOperation(store_op);
      if (!lhlo_op) continue;

      std::vector<memref::StoreOp> store_ops_in_fusion =
          collectStoreToSameMemref(store_op);

      if (!failed(inlineFuseLhloOp(rewriter, parallel_op, lhlo_op, store_op,
                                   store_ops_in_fusion))) {
        llvm::dbgs() << "Dump lhlo after inline fusion success\n";
        lhlo_op->dump();
        llvm::dbgs() << "Dump lhlo after inline fusion success end\n";
        // TODO: Check whether it is safe to erase this op like input-inline
        rewriter.eraseOp(lhlo_op);
#if 1
        llvm::dbgs() << "************erase******************** \n";
        for (auto store_op : store_ops_in_fusion) {
          store_op.getOperation()->dump();
          rewriter.eraseOp(store_op);
        }
#else
        for (auto store_op : store_ops_in_fusion) {
          auto store_op_users =
              getValueUsersInFusionLike(store_op.getMemRef(), store_op);
          llvm::dbgs() << "************erase******************** \n";
          store_op.getOperation()->dump();
          for (auto user : store_op_users) {
            user->dump();
          }
          if (store_op_users.size() == 0) {
            rewriter.eraseOp(store_op);
          }
        }
#endif
        // cleanUnusedLhloOps(parent_block, &rewriter);
        llvm::dbgs() << "Dump after inline fusion success\n";
        parallel_op->getParentOfType<lmhlo::FusionOp>()->dump();
        llvm::dbgs() << "Dump after inline fusion success end\n";
        return success();
      }
    }

    // Since we did not clean lhlo ops in InputInlineFusionPass for kWhere
    // fusion for possible output fusion, we need to do clean here.
    cleanUnusedLhloOps(parent_block, &rewriter);
    return failure();
  }

 private:
  Operation* getStorableOperation(memref::StoreOp store_op) const {
    llvm::dbgs() << "enter getStorableOperation \n";
    // store_op.getOperation()->dump();
    Operation* lhlo_op = nullptr;
    auto users = getValueUsersInFusionLike(store_op.getMemRef(), store_op);
    for (Operation* user : users) {
      if (isa<LmhloOp>(user)) {
        auto user_operand_index = isa<lmhlo::DynamicGatherOp>(user) ? 1 : 0;
        if (isSameUnderlineBuffer(
                cast<LmhloOp>(user).getOperation()->getOperand(
                    user_operand_index),
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
            //     "More than one lhlo_op write to one Memref within one
            //     fusion");
          }
          lhlo_op = user;
        }
      }
    }
    return lhlo_op;
  }

  // Collect store ops write to the same memref and within the same fusion as
  // store_op, including store_op itself.
  std::vector<memref::StoreOp> collectStoreToSameMemref(
      memref::StoreOp store_op) const {
    llvm::dbgs() << "Enter collectStoreToSameMemref\n";
    auto parallel_op = store_op->getParentOfType<scf::ParallelOp>();
    std::vector<memref::StoreOp> store_ops;
    parallel_op->walk([&](memref::StoreOp store_op_in_fusion) {
      llvm::dbgs() << "************************************* \n";
      store_op.dump();
      store_op.getMemRef().dump();
      store_op_in_fusion.dump();
      store_op_in_fusion.getMemRef().dump();
      llvm::dbgs() << "************************************* \n";

      if (store_op_in_fusion.getMemRef() == store_op.getMemRef()) {
        store_ops.push_back(store_op_in_fusion);
      }
    });
    llvm::dbgs() << "Exit collectStoreToSameMemref\n";
    return store_ops;
  }

  LogicalResult inlineFuseLhloOp(
      PatternRewriter& rewriter, scf::ParallelOp parallel_op,
      Operation* consumer, memref::StoreOp store_op,
      std::vector<memref::StoreOp>& store_ops) const {
    llvm::dbgs() << "inlineFuseLhloOp dump\n";
    consumer->dump();
    llvm::dbgs() << "inlineFuseLhloOp dump end\n";
    if (miscFuseHelper<lmhlo::DynamicReshapeOp>(rewriter, parallel_op, consumer,
                                                store_op, store_ops) ||
        miscFuseHelper<lmhlo::DynamicGatherOp>(rewriter, parallel_op, consumer,
                                               store_op, store_ops) ||
        miscFuseHelper<lmhlo::DynamicBroadcastInDimOp>(
            rewriter, parallel_op, consumer, store_op, store_ops) ||
        miscFuseHelper<lmhlo::SelectOp>(rewriter, parallel_op, consumer,
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

constexpr unsigned c_MAX_ITERATION = 4 * 1000;

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

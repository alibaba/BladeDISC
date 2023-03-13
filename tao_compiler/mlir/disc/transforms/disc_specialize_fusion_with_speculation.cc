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

// This file implements the logic for specializing the fusion kernel with
// speculation.

#include "lhlo/IR/lhlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"       // TF:local_config_mlir
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/codegen_utils.h"
#include "mlir/disc/transforms/fusion_utils.h"
#include "mlir/disc/transforms/placement_utils.h"
#include "tensorflow/core/util/env_var.h"
#include "utils/codegen_utils.h"

namespace mlir {
namespace disc_ral {
namespace {

using disc_shape::SymbolicDimOp;
using lmhlo::DynamicBroadcastInDimOp;
using lmhlo::FusionOp;

bool IsCandidateBroadcastOp(Operation* op) {
  auto broadcast_op = dyn_cast_or_null<DynamicBroadcastInDimOp>(op);
  if (!broadcast_op) return false;

  Value operand = op->getOperand(0);
  MemRefType operand_tp = operand.getType().dyn_cast<MemRefType>();
  Value result = op->getOperand(2);
  MemRefType result_tp = result.getType().dyn_cast<MemRefType>();

  if (!operand_tp || !result_tp) return false;

  // we do not do speculation if the output of a broadcast op is also the output
  // of the fusion patten.
  // TODO(disc): remove this constraint.
  FusionOp fusion_op = op->getParentOfType<FusionOp>();
  FusionPatternBase fusion_pattern{fusion_op};

  Value broadcast_result = op->getOperand(2);
  auto& results = fusion_pattern.getResults();
  if (llvm::find(results, broadcast_result) != results.end()) {
    return false;
  }

  // Return false if the ranks are mismatch.
  if (operand_tp.getRank() != result_tp.getRank()) {
    return false;
  }

  // Return false if the shapes of input and output are not compatible.
  for (auto&& e : llvm::zip(operand_tp.getShape(), result_tp.getShape())) {
    auto lhs = std::get<0>(e);
    auto rhs = std::get<1>(e);
    if (lhs != rhs) return false;
  }

  return true;
}

bool HasCandidateBroadcastOp(FusionOp fusion_op) {
  for (auto& block : fusion_op.getRegion()) {
    for (auto& op : block) {
      if (IsCandidateBroadcastOp(&op)) {
        return true;
      }
    }
  }
  return false;
}

struct ShapeConstraintIRCloneContext {
  IRMapping valueMapping;
  DenseMap<Value, SmallVector<SymbolicDimOp>> value2Symbols;
  DenseMap<SymbolicDimOp, SymbolicDimOp> symbolMapping;
  SymbolicDimMgr* mgr = nullptr;
};

FusionOp cloneFusion(OpBuilder& b, FusionOp op,
                     ShapeConstraintIRCloneContext* ctx = nullptr) {
  if (ctx == nullptr) {
    return dyn_cast<FusionOp>(b.clone(*op.getOperation()));
  }

  SymbolTable& table = ctx->mgr->symbolTable();
  DenseSet<SymbolicDimOp> originalSymbols;
  // 1, create a local view of original (dynamic shape) buffers.
  for (Operation& op : op.getRegion().front()) {
    for (Value operand : op.getOperands()) {
      auto ty = operand.getType().dyn_cast<MemRefType>();
      // skip staitc shape buffers.
      if (!ty || ty.hasStaticShape()) continue;
      if (ctx->valueMapping.lookupOrNull(operand)) continue;
      auto symbols = getMemRefValueSymbolicDimRefs(operand);
      if (!symbols) continue;
      ctx->valueMapping.map(operand,
                            createViewLike(b, op.getLoc(), operand, operand));
      auto& symbolOps = ctx->value2Symbols[operand];
      for (const auto& sym : *symbols) {
        auto symOp = table.lookup<SymbolicDimOp>(sym.getValue());
        assert(symOp);
        originalSymbols.insert(symOp);
        symbolOps.push_back(symOp);
      }
    }
  }
  // 2, clone symbol group
  auto status = ctx->mgr->cloneSymbolGroup(originalSymbols, ctx->symbolMapping);
  assert(!failed(status));

  // 3, attach new symbol attrs for the cloned operands.
  for (const auto& it : ctx->valueMapping.getValueMap()) {
    auto symIt = ctx->value2Symbols.find(it.first);
    assert(symIt != ctx->value2Symbols.end());
    Operation* definingOp = it.second.getDefiningOp();
    assert(definingOp != nullptr);
    SmallVector<Attribute> newAttrs;
    SmallVector<SymbolicDimOp> newSymbols;
    for (SymbolicDimOp sym : symIt->second) {
      SymbolicDimOp newSymbol = ctx->symbolMapping[sym];
      newSymbols.push_back(newSymbol);
      newAttrs.push_back(FlatSymbolRefAttr::get(newSymbol));
    }
    auto symbolicShapeAttr = ArrayAttr::get(op->getContext(), newAttrs);
    StringRef attrName = disc_shape::SymbolicDimOp::getSymbolicDimAttrName();
    definingOp->setAttr(attrName, symbolicShapeAttr);
    ctx->value2Symbols[it.second] = std::move(newSymbols);
  }

  return dyn_cast<FusionOp>(b.clone(*op.getOperation(), ctx->valueMapping));
}

FusionOp cloneWithBroadcastSimplifying(
    OpBuilder& b, FusionOp fusion_op,
    SmallVectorImpl<Operation*>& broadcast_ops,
    ShapeConstraintIRCloneContext* ctx = nullptr) {
  FusionOp cloned = cloneFusion(b, fusion_op, ctx);

  // Collects all candidate broadcast ops inside the fusion op.
  cloned.walk([&](DynamicBroadcastInDimOp op) {
    if (IsCandidateBroadcastOp(op)) {
      broadcast_ops.push_back(op);
    }
  });

  for (Operation* op : broadcast_ops) {
    Value operand = op->getOperand(0);
    Value result = op->getOperand(2);
    for (Operation* user : llvm::to_vector<4>(result.getUsers())) {
      if (user == op) continue;
      // Skip the user that not inside the same block.
      if (op->getBlock() != user->getBlock()) {
        continue;
      }
      user->replaceUsesOfWith(result, operand);
      if (ctx != nullptr) {
        auto lhsIt = ctx->value2Symbols.find(result);
        auto rhsIt = ctx->value2Symbols.find(operand);
        if (lhsIt == ctx->value2Symbols.end() ||
            rhsIt == ctx->value2Symbols.end())
          continue;

        assert(lhsIt->second.size() == rhsIt->second.size());
        for (const auto& z : llvm::zip(lhsIt->second, rhsIt->second))
          ctx->mgr->mapSymbolicDimEqual(std::get<0>(z), std::get<1>(z));
      }
    }
  }
  return cloned;
}

Operation* GetCandidateRowReduceOp(FusionOp fusion_op) {
  for (Block& block : fusion_op.getRegion().getBlocks()) {
    for (Operation& op : block) {
      // All row reduce op should have the same shape, thus we can return any
      // of them.
      if (isRank2RowReduction(&op)) {
        return &op;
      }
    }
  }
  return nullptr;
}

Operation* GetCandidateColReduceOp(FusionOp fusion_op) {
  for (Block& block : fusion_op.getRegion().getBlocks()) {
    for (Operation& op : block) {
      // All col reduce op should have the same shape, thus we can return any
      // of them.
      if (isRank2ColReduction(&op)) {
        return &op;
      }
    }
  }
  return nullptr;
}

struct DiscSpecializeFusionWithSpeculationPass
    : public DiscSpecializeFusionWithSpeculationPassBase<
          DiscSpecializeFusionWithSpeculationPass> {
  DiscSpecializeFusionWithSpeculationPass(int core_count,
                                          int max_threads_per_core) {
    core_count_ = core_count;
    max_threads_per_core_ = max_threads_per_core;
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<scf::SCFDialect, memref::MemRefDialect>();
  }

  void DoBroadcastSpeculation(FusionOp fusion_op) {
    // We only do specialization for fusion op having candidate broadcast ops.
    if (!HasCandidateBroadcastOp(fusion_op)) {
      return;
    }

    // basic logic is shown as below:
    // %pred = ...
    // if (%pred) {
    //   call %simplified_fusion_func(...)
    // } else {
    //   call %original_fusion_func(...)
    // }
    Value pred;
    OpBuilder b(fusion_op);
    Location loc = fusion_op.getLoc();

    // Clone the fusion op and mark all candidate broadcast ops within the
    // fusion op.
    ShapeConstraintIRCloneContext ctx;
    ctx.mgr = symbolMgr.get();
    SmallVector<Operation*, 4> broadcast_ops;
    FusionOp cloned = cloneWithBroadcastSimplifying(
        b, fusion_op, broadcast_ops, useShapeConstraintIR() ? &ctx : nullptr);
    addFusionTag(b, cloned, "no_ib");

    // Generate the predition.
    for (Operation* op : broadcast_ops) {
      Value operand = op->getOperand(0);
      Value result = op->getOperand(2);
      int rank = operand.getType().cast<MemRefType>().getRank();
      for (int i = 0; i < rank; ++i) {
        Value lhs = b.create<memref::DimOp>(loc, operand, i);
        Value rhs = b.create<memref::DimOp>(loc, result, i);
        Value eq =
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, lhs, rhs);
        pred = (pred ? b.create<arith::AndIOp>(loc, eq, pred).getResult() : eq);
      }
      op->erase();
    }

    assert(!broadcast_ops.empty());

    auto if_op = b.create<scf::IfOp>(loc, llvm::None, pred, true);
    Block* then_block = &if_op.getThenRegion().getBlocks().front();
    Block* else_block = &if_op.getElseRegion().getBlocks().front();
    cloned.getOperation()->moveBefore(then_block, then_block->begin());
    fusion_op.getOperation()->moveBefore(else_block, else_block->begin());

    DenseMap<Value, Value> viewMap;
    if (useShapeConstraintIR()) {
      OpBuilder viewBuilder(cloned);
      // build symbolicDimOp to its corresponding SSA value map
      DenseMap<SymbolicDimOp, Value> symbolicDim2SSAValue;
      for (Operation& op : cloned.getRegion().front()) {
        for (Value operand : op.getOperands()) {
          auto it = ctx.value2Symbols.find(operand);
          if (it == ctx.value2Symbols.end()) continue;
          for (const auto& en : llvm::enumerate(it->second)) {
            SymbolicDimOp rootSym = ctx.mgr->getRootSymbolicDim(en.value());
            if (en.value() != rootSym) continue;
            if (rootSym.isDynamic()) {
              symbolicDim2SSAValue[rootSym] = viewBuilder.create<memref::DimOp>(
                  op.getLoc(), operand, en.index());
            } else {
              symbolicDim2SSAValue[rootSym] =
                  viewBuilder.create<arith::ConstantIndexOp>(
                      op.getLoc(), rootSym.getDimSize());
            }
          }
        }
      }

      for (Operation& op : cloned.getRegion().front()) {
        for (Value operand : op.getOperands()) {
          if (viewMap.find(operand) != viewMap.end()) continue;
          auto it = ctx.value2Symbols.find(operand);
          if (it == ctx.value2Symbols.end()) continue;
          SmallVector<int64_t> newShape;
          SmallVector<Value> newShapeValues;
          SmallVector<Attribute> newAttrs;
          for (SymbolicDimOp sym : it->second) {
            SymbolicDimOp rootSym = ctx.mgr->getRootSymbolicDim(sym);
            newAttrs.push_back(FlatSymbolRefAttr::get(rootSym));
            newShapeValues.push_back(symbolicDim2SSAValue[rootSym]);
            newShape.push_back(rootSym.isDynamic() ? ShapedType::kDynamic
                                                   : rootSym.getDimSize());
          }
          auto oldType = operand.getType().cast<MemRefType>();
          auto newType =
              MemRefType::get(newShape, oldType.getElementType(),
                              oldType.getLayout(), oldType.getMemorySpace());
          // TODO(disc): handle mismatch type case
          if (newType != oldType) continue;
          viewMap[operand] = CastMemRefTo(viewBuilder, op.getLoc(), operand,
                                          newType, newShapeValues);
          auto symbolicShapeAttr = ArrayAttr::get(op.getContext(), newAttrs);
          StringRef attrName = SymbolicDimOp::getSymbolicDimAttrName();
          viewMap[operand].getDefiningOp()->setAttr(attrName,
                                                    symbolicShapeAttr);
        }
      }
    } else {
      SmallVector<Operation*, 4> op_list;
      for (Operation& op : cloned.getRegion().front()) op_list.push_back(&op);
      OpListShapeAnalysis op_list_shape_analysis(op_list);
      for (Operation* op : op_list) {
        for (Value operand : op->getOperands()) {
          if (viewMap.find(operand) != viewMap.end()) continue;
          Value leader =
              op_list_shape_analysis.GetLeaderValueWithSameShape(operand);
          if (!leader || leader == operand) continue;
          // TODO(disc): handle mismatch type case
          auto leaderTy = leader.getType().dyn_cast<MemRefType>();
          auto operandTy = operand.getType().dyn_cast<MemRefType>();
          if (!leaderTy || !operandTy ||
              leaderTy.getRank() != operandTy.getRank())
            continue;
          bool sameShape = true;
          for (auto&& en :
               llvm::zip(leaderTy.getShape(), operandTy.getShape())) {
            if (std::get<0>(en) != std::get<1>(en)) {
              sameShape = false;
              break;
            }
          }
          if (!sameShape) continue;
          OpBuilder viewBuilder(cloned);
          viewMap[operand] =
              createViewLike(viewBuilder, op->getLoc(), operand, leader);
        }
      }
    }

    cloned.walk([&](Operation* op) {
      for (auto& it : viewMap) op->replaceUsesOfWith(it.first, it.second);
    });
  }

  void DoRowReductionSpeculation(FusionOp fusion_op) {
    FusionType fusion_type = getFusionType(fusion_op.getOperation());
    if (fusion_type != FusionType::kRowReduction &&
        fusion_type != FusionType::kStitch) {
      return;
    }

    // We only do specialization fusion op on GPU a.t.m.
    if (!placement_utils::isGpuMhlo(fusion_op)) {
      return;
    }

    // Already have a hint
    if (fusion_op->getAttrOfType<IntegerAttr>(kRowReductionScheduleHint)) {
      return;
    }

    Operation* reduce_op = GetCandidateRowReduceOp(fusion_op);
    assert(reduce_op != nullptr);

    OpBuilder b(fusion_op);
    Location loc = fusion_op.getLoc();
    FusionOp cloned = dyn_cast<FusionOp>(b.clone(*fusion_op.getOperation()));

    Value operand = reduce_op->getOperand(0);
    // TODO(disc): Use 256 as default block size; turn this number for
    // different shapes
    int block_size = kThreadsRowReduction;
    Value col_size = b.create<memref::DimOp>(loc, operand, 1);
    Value pred;

    // TODO: this feature will be experimental in the first release, and will be
    // set as default in the near future after evaluated on benchmarks.
    if (isMemIntensiveOptExperimentalEnabled() && core_count_ != -1 &&
        max_threads_per_core_ != -1) {
      // When the number of rows is small or the number of cols is large, we
      // use one-block-one-row schedule. Specifically, we use one-block-one-row
      // schedule for the following conditions (otherwise use one-warp-one-row
      // schedule):
      //   1. row < max-warps-per-wave / 4
      //   2. row < max-warps-per-wave / 2 & col >= 1024
      //   3. row >= max-warps-per-wave / 2 & cols >= 512
      // Note the block-size is 256 currently.
      int64_t max_warps_per_wave =
          max_threads_per_core_ / block_size * core_count_ / kWarpSize;
      Value row_size = b.create<memref::DimOp>(loc, operand, 0);
      Value col_size = b.create<memref::DimOp>(loc, operand, 1);
      Value cond1 = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, row_size,
          b.create<arith::ConstantIndexOp>(loc, max_warps_per_wave / 4));
      Value cond2 = b.create<arith::AndIOp>(
          loc,
          b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::slt, row_size,
              b.create<arith::ConstantIndexOp>(loc, max_warps_per_wave / 2)),
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, col_size,
                                  b.create<arith::ConstantIndexOp>(loc, 1024)));
      Value cond3 = b.create<arith::AndIOp>(
          loc,
          b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::sge, row_size,
              b.create<arith::ConstantIndexOp>(loc, max_warps_per_wave / 2)),
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, col_size,
                                  b.create<arith::ConstantIndexOp>(loc, 512)));
      pred = b.create<arith::OrIOp>(loc, cond1, cond2);
      pred = b.create<arith::OrIOp>(loc, pred, cond3);
    } else {
      // Default schedule selection policy:
      //   use schedule 1 if col size > `kRowReductionScheduleTurningSize`;
      //   use schedule 2 otherwise.
      Value ref_size = b.create<arith::ConstantIndexOp>(
          loc, kRowReductionScheduleTurningSize);
      pred = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, col_size,
                                     ref_size);
    }

    auto if_op = b.create<scf::IfOp>(loc, llvm::None, pred, true);

    auto first_schedule = b.getIntegerAttr(b.getIntegerType(32), 1);
    auto second_schedule = b.getIntegerAttr(b.getIntegerType(32), 2);
    auto num_thread_attr = b.getIntegerAttr(b.getIntegerType(32), block_size);
    fusion_op->setAttr(kThreadPerBlockHint, num_thread_attr);
    fusion_op->setAttr(kRowReductionScheduleHint, first_schedule);
    // one block one row
    addFusionTag(b, fusion_op, "1b1r");
    cloned->setAttr(kThreadPerBlockHint, num_thread_attr);
    cloned->setAttr(kRowReductionScheduleHint, second_schedule);
    // one warp one row
    addFusionTag(b, cloned, "1w1r");

    Block* then_block = &if_op.getThenRegion().getBlocks().front();
    Block* else_block = &if_op.getElseRegion().getBlocks().front();
    fusion_op.getOperation()->moveBefore(then_block, then_block->begin());
    cloned.getOperation()->moveBefore(else_block, else_block->begin());
  }

  // TODO(feiwen): add more kTileW=8/kTileH pairs by if/elseif/else
  void DoColReductionSpeculation(FusionOp fusion_op) {
    // We only do specialization fusion op on GPU a.t.m.
    bool use_new = false;
    static const char* env = getenv("NEW_COL");
    if (env != nullptr) {
      use_new = std::string(env) == "1" || std::string(env) == "2" ||
                std::string(env) == "4" || std::string(env) == "8";
    }

    if (!placement_utils::isGpuMhlo(fusion_op)) {
      return;
    }

    if ((core_count_ == -1 || max_threads_per_core_ == -1) && (!use_new)) {
      // Do not know about device information.
      return;
    }

    // Already have a hint
    if (fusion_op->getAttrOfType<IntegerAttr>(kColReductionScheduleHint))
      return;

    Operation* reduce_op = nullptr;
    if (!(reduce_op = GetCandidateColReduceOp(fusion_op))) {
      return;
    }

    // Col reduction schedule selection policy: if blocks > 80, use schedule
    // kTileW=8/kTileH=32; or use schedule kTileW=8/kTileH=8.
    OpBuilder b(fusion_op);
    Location loc = fusion_op.getLoc();
    FusionOp cloned = dyn_cast<FusionOp>(b.clone(*fusion_op.getOperation()));

    Value operand = reduce_op->getOperand(0);
    Value row_size = b.create<memref::DimOp>(loc, operand, 0);
    Value col_size = b.create<memref::DimOp>(loc, operand, 1);
    Value matrix_size = b.create<arith::MulIOp>(loc, row_size, col_size);
    int thread_per_block = kThreadsRowReduction;
    Value cur_threads = b.create<arith::ConstantIndexOp>(loc, thread_per_block);
    // b.create<arith::ConstantIndexOp>(loc, max_threads_per_block_);
    Value cur_blocks =
        b.create<arith::CeilDivSIOp>(loc, matrix_size, cur_threads);
    Value ref_blocks = b.create<arith::ConstantIndexOp>(loc, core_count_);

    Value pred = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                         cur_blocks, ref_blocks);

    auto if_op = b.create<scf::IfOp>(loc, llvm::None, pred, true);

    auto w8_h32_schedule =
        b.getIntegerAttr(b.getIntegerType(32), DISC_TILE_W8_H32);
    auto w8_h16_schedule =
        b.getIntegerAttr(b.getIntegerType(32), DISC_TILE_W8_H16);
    auto num_thread_full_attr =
        b.getIntegerAttr(b.getIntegerType(32), kThreadsRowReduction);
    auto num_thread_half_attr =
        b.getIntegerAttr(b.getIntegerType(32), kThreadsRowReduction / 2);
    fusion_op->setAttr(kThreadPerBlockHint, num_thread_full_attr);
    fusion_op->setAttr(kColReductionScheduleHint, w8_h32_schedule);
    // use 8*32 tile if block# >= SM#
    addFusionTag(b, fusion_op, "8w32h");
    cloned->setAttr(kThreadPerBlockHint, num_thread_half_attr);
    cloned->setAttr(kColReductionScheduleHint, w8_h16_schedule);
    // one 8*16 tile if block# < SM#
    addFusionTag(b, cloned, "8w16h");

    Block* then_block = &if_op.getThenRegion().getBlocks().front();
    Block* else_block = &if_op.getElseRegion().getBlocks().front();
    fusion_op.getOperation()->moveBefore(then_block, then_block->begin());
    cloned.getOperation()->moveBefore(else_block, else_block->begin());
  }

  void DoVectorizeOrTileSpeculation(FusionOp fusion_op) {
    if (!placement_utils::isGpuMhlo(fusion_op)) {
      return;
    }

    if (core_count_ == -1 || max_threads_per_core_ == -1) {
      // Do not know about device information.
      return;
    }

    // Already have a hint
    if (fusion_op->getAttrOfType<IntegerAttr>(kVectorizeOrTileHint)) return;

    bool enable_mem_intensive_opt_expreimental =
        isMemIntensiveOptExperimentalEnabled();
    FusionType fusion_type = getFusionType(fusion_op.getOperation());
    if (fusion_type != FusionType::kLoop &&
        fusion_type != FusionType::kRowReduction &&
        (fusion_type != FusionType::kStitch ||
         (fusion_type == FusionType::kStitch &&
          enable_mem_intensive_opt_expreimental))) {
      // TODO: support tile optimization for kStitch fusion when
      // `DISC_MEM_INTENSIVE_OPT_EXPERIMENTAL` is `true`.
      return;
    }

    // When 'MemIntensiveOptExperimental' is enabled, the vecotrization of
    // concatenate operator will peform a bad case on bert model.
    // Skip optimization of concatenate operator here.
    FusionPatternBase fusion_pattern(fusion_op);
    bool contain_concatenate = false;
    // Currently, skip `enable_mem_intensive_opt_expreimental` if the fusion
    // contains concatenate operator.
    for (auto op : fusion_pattern.getOpList()) {
      if (isa<lmhlo::ConcatenateOp>(op)) {
        contain_concatenate = true;
        break;
      }
    }
    enable_mem_intensive_opt_expreimental &= !contain_concatenate;

    // TODO: aware of row-reduction hints.

    // Default vectorization/tiling policy:
    //
    // 1) No vectorization/Tiling.
    //     - dominanted by column reduction (codegen support now). TODO: add
    //       vectorization/tiling support.
    //     - dominanted by row-reduction: row-number % kVectorizeOrTileSize
    //       != 0, or row-number < max-rows-per-thread-wave.
    //     - dominanted by element-wise: element-number % kVectorizeOrTileSize
    //       != 0, or element-number < max-threads-per-wave.
    // 2) Otherwise apply vectorization/tiling with width of
    //    `kVectorizeOrTileSize`, which is 2 currently.

    OpBuilder b(fusion_op);
    Location loc = fusion_op.getLoc();

    Value pred;
    int max_threads_per_wave = core_count_ * max_threads_per_core_;
    int vector_size = kVectorizeOrTileSize;
    if (fusion_type == FusionType::kRowReduction ||
        fusion_type == FusionType::kStitch) {
      Operation* dominant_equivalent_op = GetCandidateRowReduceOp(fusion_op);
      auto block_size = getThreadPerBlock(fusion_op.getOperation());

      int rowred_schedule =
          getRowReductionScheduleHint(fusion_op.getOperation());

      auto row_per_block = (rowred_schedule == DISC_WARP_WISE_ROW_REDUCE)
                               ? block_size / kWarpSize
                               : 1;
      auto max_rows_per_wave =
          max_threads_per_wave / block_size * row_per_block;
      Value threshold =
          b.create<arith::ConstantIndexOp>(loc, max_rows_per_wave);
      Value operand = dominant_equivalent_op->getOperand(0);
      // #out-element is row numbers.
      Value out_element_number = b.create<memref::DimOp>(loc, operand, 0);

      Value larger = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                             out_element_number, threshold);
      Value divisible = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq,
          b.create<arith::RemUIOp>(
              loc, out_element_number,
              b.create<arith::ConstantIndexOp>(loc, vector_size)),
          b.create<arith::ConstantIndexOp>(loc, 0));
      pred = b.create<arith::AndIOp>(loc, larger, divisible);
    } else if (fusion_type == FusionType::kLoop) {
      Operation* dominant_equivalent_op = fusion_pattern.getRootOps().back();
      Value out_element_number =
          emitNumElementsComputation(b, loc, dominant_equivalent_op);
      if (enable_mem_intensive_opt_expreimental) {
        // Maximize the vector-size according to data type of all outputs. The
        // maximum vector-size is 8 (128 / 16).
        auto& results = fusion_pattern.getResults();
        const int gpu_vector_width = 128;
        int min_bits = gpu_vector_width;
        for (auto result : results) {
          auto memref_ty = result.getType().cast<MemRefType>();
          int byte_width = memref_ty.getElementTypeBitWidth();
          min_bits = std::min(byte_width, min_bits);
        }
        // The minimum bit-width for vector-size calculation is 16.
        min_bits = std::max(min_bits, 16);
        assert(gpu_vector_width % min_bits == 0);
        vector_size = gpu_vector_width / min_bits;
      }
      Value divisible = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq,
          b.create<arith::RemUIOp>(
              loc, out_element_number,
              b.create<arith::ConstantIndexOp>(loc, vector_size)),
          b.create<arith::ConstantIndexOp>(loc, 0));
      if (enable_mem_intensive_opt_expreimental) {
        pred = divisible;
      } else {
        Value threshold =
            b.create<arith::ConstantIndexOp>(loc, max_threads_per_wave);
        Value larger = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                               out_element_number, threshold);
        pred = b.create<arith::AndIOp>(loc, larger, divisible);
      }
    } else {
      // Either a column reduction dominanted fusion, or a non-fusion op.
      return;
    }

    auto if_op = b.create<scf::IfOp>(loc, llvm::None, pred, true);

    // Vectorization/tiling branch.
    auto vec_tile = b.getIntegerAttr(b.getIntegerType(32), vector_size);
    fusion_op->setAttr(kVectorizeOrTileHint, vec_tile);
    Block* then_block = &if_op.getThenRegion().getBlocks().front();
    fusion_op.getOperation()->moveBefore(then_block, then_block->begin());

    // Non-vectorization/tiling branch.
    FusionOp cloned = dyn_cast<FusionOp>(b.clone(*fusion_op.getOperation()));
    auto no_vec_tile = b.getIntegerAttr(b.getIntegerType(32), 1);
    cloned->setAttr(kVectorizeOrTileHint, no_vec_tile);
    Block* else_block = &if_op.getElseRegion().getBlocks().front();
    cloned.getOperation()->moveBefore(else_block, else_block->begin());

    // Add tag for vectorized op after it is cloned.
    addFusionTag(b, fusion_op, "Vec" + std::to_string(vector_size));
  }

  void Speculator(
      std::function<void(DiscSpecializeFusionWithSpeculationPass*, FusionOp)>
          cb) {
    // Collects the fusion ops first since following rewriter may insert new
    // fusion ops.
    SmallVector<FusionOp, 4> fusion_ops;
    getOperation().walk(
        [&](FusionOp fusion_op) { fusion_ops.emplace_back(fusion_op); });

    for (FusionOp fusion_op : fusion_ops) {
      FusionPatternBase fusion_pattern(fusion_op);
      auto root_ops = fusion_pattern.getRootOps();
      // Skip dead fusions
      if (!root_ops.empty()) {
        cb(this, fusion_op);
      }
    };
  }

  void runOnOperation() override {
    if (useShapeConstraintIR()) {
      func::FuncOp func = getOperation();
      // skip shape constraint graph
      if (func.getName() ==
          SymbolicDimMgr::getShapeConstraintGraphFunctionName())
        return;

      auto m = func->getParentOfType<ModuleOp>();
      symbolMgr.reset(new SymbolicDimMgr(m));
      if (failed(symbolMgr->load())) {
        getOperation()->emitError() << "fail to load shape constraint IR\n";
        signalPassFailure();
        return;
      }
    }

    // Stage #1: broadcast simplifier with speculation.
    Speculator(
        &DiscSpecializeFusionWithSpeculationPass::DoBroadcastSpeculation);

    // Stage #2: speculation of column reduce op.
    Speculator(
        &DiscSpecializeFusionWithSpeculationPass::DoColReductionSpeculation);

    // Stage #3: speculation of row reduce op.
    // We do row-reduction speculation before vectorization/tiling because TLP
    // is usually more important than ILP on GPU.
    Speculator(
        &DiscSpecializeFusionWithSpeculationPass::DoRowReductionSpeculation);

    // Stage #4: speculation of vectorization/tiling.
    Speculator(
        &DiscSpecializeFusionWithSpeculationPass::DoVectorizeOrTileSpeculation);

    if (useShapeConstraintIR()) {
      if (failed(symbolMgr->save())) {
        getOperation()->emitError() << "fail to load shape constraint IR\n";
        signalPassFailure();
        return;
      }
    }
  }

  std::shared_ptr<SymbolicDimMgr> symbolMgr;
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createDiscSpecializeFusionWithSpeculationPass(int core_count,
                                              int max_threads_per_core) {
  return std::make_unique<DiscSpecializeFusionWithSpeculationPass>(
      core_count, max_threads_per_core);
}

}  // namespace disc_ral
}  // namespace mlir

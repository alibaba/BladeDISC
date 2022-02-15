/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This file implements the logic for specializing the fusion kernel with
// speculation.

#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir-hlo/utils/codegen_utils.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"       // TF:local_config_mlir
#include "tensorflow/compiler/mlir/disc/transforms/PassDetail.h"
#include "tensorflow/compiler/mlir/disc/transforms/codegen_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/fusion_utils.h"
#include "tensorflow/compiler/mlir/disc/transforms/placement_utils.h"
#include "tensorflow/core/util/env_var.h"

namespace mlir {
namespace disc_ral {
namespace {

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
  for (auto& block : fusion_op.region()) {
    for (auto& op : block) {
      if (IsCandidateBroadcastOp(&op)) {
        return true;
      }
    }
  }
  return false;
}

Value createViewLike(OpBuilder& b, Location loc, Value from, Value to) {
  SmallVector<Value> toShape = getShapeValues(&b, to);
  auto toType = to.getType().cast<MemRefType>();
  auto fromType = from.getType().cast<MemRefType>();
  auto targetType =
      MemRefType::get(toType.getShape(), fromType.getElementType(),
                      toType.getLayout(), toType.getMemorySpace());
  return CastMemRefTo(b, loc, from, targetType, toShape);
}

FusionOp cloneWithBroadcastSimplifying(
    OpBuilder& b, FusionOp fusion_op,
    SmallVectorImpl<Operation*>& broadcast_ops) {
  FusionOp cloned = dyn_cast<FusionOp>(b.clone(*fusion_op.getOperation()));

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
    }
  }
  return cloned;
}

Operation* GetCandidateRowReduceOp(FusionOp fusion_op) {
  for (Block& block : fusion_op.region().getBlocks()) {
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
  for (Block& block : fusion_op.region().getBlocks()) {
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
  DiscSpecializeFusionWithSpeculationPass(int cc_major, int cc_minor) {
    cc_major_ = cc_major;
    cc_minor_ = cc_minor;
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
    SmallVector<Operation*, 4> broadcast_ops;
    FusionOp cloned =
        cloneWithBroadcastSimplifying(b, fusion_op, broadcast_ops);
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
    SmallVector<Operation*, 4> op_list;
    for (Operation& op : cloned.region().front()) op_list.push_back(&op);
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
        for (auto&& en : llvm::zip(leaderTy.getShape(), operandTy.getShape())) {
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

    // Row reduction schedule selection policy: if col size > 512, use schedule
    // 1; also use schedule 2. The 512 is an empirical value.
    OpBuilder b(fusion_op);
    Location loc = fusion_op.getLoc();
    FusionOp cloned = dyn_cast<FusionOp>(b.clone(*fusion_op.getOperation()));

    Value operand = reduce_op->getOperand(0);
    Value col_size = b.create<memref::DimOp>(loc, operand, 1);
    Value ref_size =
        b.create<arith::ConstantIndexOp>(loc, kRowReductionScheduleTurningSize);
    Value pred = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                         col_size, ref_size);

    auto if_op = b.create<scf::IfOp>(loc, llvm::None, pred, true);

    // TODO(disc): Use 256 as default block size; turn this number for
    // different shapes
    int thread_per_block = kThreadsRowReduction;
    auto first_schedule = b.getIntegerAttr(b.getIntegerType(32), 1);
    auto second_schedule = b.getIntegerAttr(b.getIntegerType(32), 2);
    auto num_thread_attr =
        b.getIntegerAttr(b.getIntegerType(32), thread_per_block);
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
    if (!placement_utils::isGpuMhlo(fusion_op)) {
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
    int thread_per_block = 256;
    Value cur_threads = b.create<arith::ConstantIndexOp>(loc, thread_per_block);
    Value cur_blocks =
        b.create<arith::CeilDivSIOp>(loc, matrix_size, cur_threads);
    int sm_num;
    auto thread_number_info =
        archToGPUThreadNumber.find({cc_major_, cc_minor_});
    if (thread_number_info != archToGPUThreadNumber.end()) {
      auto info = thread_number_info->second;
      sm_num = info.first;
    } else {
      sm_num = 80;  // Default is the data of V100.
    }
    Value ref_blocks = b.create<arith::ConstantIndexOp>(loc, sm_num);

    Value pred = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                         cur_blocks, ref_blocks);

    auto if_op = b.create<scf::IfOp>(loc, llvm::None, pred, true);

    auto w8_h32_schedule =
        b.getIntegerAttr(b.getIntegerType(32), DISC_TILE_W8_H32);
    auto w8_h16_schedule =
        b.getIntegerAttr(b.getIntegerType(32), DISC_TILE_W8_H16);
    auto num_thread_256_attr = b.getIntegerAttr(b.getIntegerType(32), 256);
    auto num_thread_128_attr = b.getIntegerAttr(b.getIntegerType(32), 128);
    fusion_op->setAttr(kThreadPerBlockHint, num_thread_256_attr);
    fusion_op->setAttr(kColReductionScheduleHint, w8_h32_schedule);
    // use 8*32 tile if block# >= SM#
    addFusionTag(b, fusion_op, "8w32h");
    cloned->setAttr(kThreadPerBlockHint, num_thread_128_attr);
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

    // Already have a hint
    if (fusion_op->getAttrOfType<IntegerAttr>(kVectorizeOrTileHint)) return;

    FusionType fusion_type = getFusionType(fusion_op.getOperation());
    if (fusion_type != FusionType::kLoop &&
        fusion_type != FusionType::kRowReduction &&
        fusion_type != FusionType::kStitch) {
      return;
    }

    // Vectorization/tiling policy:
    //
    // 1) No vectorization/Tiling.
    //     - dominanted by column reduction (codegen support now). TODO: add
    //       vectorization/tiling support.
    //     - dominanted by row-reduction: row-number % kVectorizeOrTileSize
    //       != 0, or row-number < max-threads-per-wave.
    //     - dominanted by element-wise: element-number % kVectorizeOrTileSize
    //       != 0, or element-number < max-threads-per-wave.
    // 2) Otherwise apply vectorization/tiling with width of
    //    `kVectorizeOrTileSize`, which is 2 currently.

    int max_threads_per_wave;
    auto thread_number_info =
        archToGPUThreadNumber.find({cc_major_, cc_minor_});
    if (thread_number_info != archToGPUThreadNumber.end()) {
      auto info = thread_number_info->second;
      max_threads_per_wave = info.first * info.second;
    } else {
      max_threads_per_wave = 40 * 1024;  // Default is the data of T4.
    }

    OpBuilder b(fusion_op);
    Location loc = fusion_op.getLoc();

    Value out_element_number;
    Value threshold;
    if (fusion_type == FusionType::kRowReduction ||
        fusion_type == FusionType::kStitch) {
      Operation* dominant_equivalent_op = GetCandidateRowReduceOp(fusion_op);
      auto block_size = getThreadPerBlock(fusion_op.getOperation());

      int rowred_schedule =
          getRowReductionScheduleHint(fusion_op.getOperation());

      auto row_per_block = (rowred_schedule == DISC_WARP_WISE_ROW_REDUCE)
                               ? block_size / kWarpSize
                               : 1;
      auto threshold_val = max_threads_per_wave / block_size * row_per_block;
      threshold = b.create<arith::ConstantIndexOp>(loc, threshold_val);
      Value operand = dominant_equivalent_op->getOperand(0);
      // #out-element is row numbers.
      out_element_number = b.create<memref::DimOp>(loc, operand, 0);
    } else if (fusion_type == FusionType::kLoop) {
      threshold = b.create<arith::ConstantIndexOp>(loc, max_threads_per_wave);
      FusionPatternBase fusion_pattern(fusion_op);
      Operation* dominant_equivalent_op = fusion_pattern.getRootOps().back();
      out_element_number = codegen_utils::emitNumElementsComputation(
          b, loc, dominant_equivalent_op);
    } else {
      // Either a column reduction dominanted fusion, or a non-fusion op.
      return;
    }

    Value larger = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                           out_element_number, threshold);
    Value is_even = b.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq,
        b.create<arith::RemUIOp>(
            loc, out_element_number,
            b.create<arith::ConstantIndexOp>(loc, kVectorizeOrTileSize)),
        b.create<arith::ConstantIndexOp>(loc, 0));
    Value pred = b.create<arith::AndIOp>(loc, larger, is_even);
    auto if_op = b.create<scf::IfOp>(loc, llvm::None, pred, true);

    // Vectorization/tiling branch.
    auto vec_tile =
        b.getIntegerAttr(b.getIntegerType(32), kVectorizeOrTileSize);
    fusion_op->setAttr(kVectorizeOrTileHint, vec_tile);
    addFusionTag(b, fusion_op,
                 "_vectile" + std::to_string(kVectorizeOrTileSize));
    Block* then_block = &if_op.getThenRegion().getBlocks().front();
    fusion_op.getOperation()->moveBefore(then_block, then_block->begin());

    // Non-vectorization/tiling branch.
    FusionOp cloned = dyn_cast<FusionOp>(b.clone(*fusion_op.getOperation()));
    auto no_vec_tile = b.getIntegerAttr(b.getIntegerType(32), 1);
    cloned->setAttr(kVectorizeOrTileHint, no_vec_tile);
    addFusionTag(b, cloned, "_no_vectile");
    Block* else_block = &if_op.getElseRegion().getBlocks().front();
    cloned.getOperation()->moveBefore(else_block, else_block->begin());
  }

  void Speculator(
      std::function<void(DiscSpecializeFusionWithSpeculationPass*, FusionOp)>
          cb) {
    // Collects the fusion ops first since following rewriter may insert new
    // fusion ops.
    SmallVector<FusionOp, 4> fusion_ops;
    getFunction().walk(
        [&](FusionOp fusion_op) { fusion_ops.emplace_back(fusion_op); });

    for (FusionOp fusion_op : fusion_ops) {
      cb(this, fusion_op);
    };
  }

  void runOnFunction() override {
    // Stage #1: broadcast simplifier with speculation
    Speculator(
        &DiscSpecializeFusionWithSpeculationPass::DoBroadcastSpeculation);

    // Stage #2: speculation of reduce op
    Speculator(
        &DiscSpecializeFusionWithSpeculationPass::DoRowReductionSpeculation);
    Speculator(
        &DiscSpecializeFusionWithSpeculationPass::DoColReductionSpeculation);

    // Stage #3: speculation of vectorization/tiling.
    Speculator(
        &DiscSpecializeFusionWithSpeculationPass::DoVectorizeOrTileSpeculation);
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>>
createDiscSpecializeFusionWithSpeculationPass(int cc_major, int cc_minor) {
  return std::make_unique<DiscSpecializeFusionWithSpeculationPass>(cc_major,
                                                                   cc_minor);
}

}  // namespace disc_ral
}  // namespace mlir

// Copyright 2023 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/disc/IR/hlo_disc_ops.h"
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/shape_utils.h"
#include "mlir/disc/utils/cycle_detector.h"

#define DEBUG_TYPE "disc-quantized-dot-merge"

namespace mlir {
namespace disc_ral {
namespace {

// Arrange the insert-point of `op`'s operands in the same block. Do not deal
// with cross-block operands.
void ArrangeOperandsInsertPointInBlock(Operation* op) {
  for (auto operand : GetAllPossibleUsedValues(op)) {
    auto operandOp = operand.getDefiningOp();
    // Note that `isBeforeInBlock` also check whether `op` and `operandOp` are
    // in the same block.
    if ((operandOp != nullptr) && !operandOp->isBeforeInBlock(op)) {
      operandOp->moveBefore(op);
      ArrangeOperandsInsertPointInBlock(operandOp);
    }
  }
}

struct QuantizedDotCluster {
  QuantizedDotCluster(Operation* op, int op_id) : leader_op_id(op_id) {
    ops.push_back(op);
    auto rhs_op_type =
        op->getOperands()[1].getType().dyn_cast<RankedTensorType>();
    // Since layout of weight is nxk, contract dimension would always be 1.
    // TODO: support more general case if necessary.
    auto contract_dim_size_ = rhs_op_type.getDimSize(1);
    if (contract_dim_size == ShapedType::kDynamic) {
      contract_dim_size = 0;
    } else {
      contract_dim_size = contract_dim_size_;
    }
  }

  // Merges `other` into this cluster, and clears `other`.
  void merge(QuantizedDotCluster& other) {
    ops.insert(ops.end(), other.ops.begin(), other.ops.end());
    other.ops.clear();
  }

  // ID of the representative node of this cluster.
  int leader_op_id;

  // Contracting dimension size of cluster
  int contract_dim_size;

  // Dot ops to be batched.
  SmallVector<Operation*> ops;
};

// The `src` cluster will be cleared if the merge is succeed.
void TryMergeQuantizedDotClusters(
    QuantizedDotCluster& dst, QuantizedDotCluster& src,
    std::unique_ptr<GraphCycles>& cycle_detector) {
  // Check cycle.
  int64_t dst_id = dst.leader_op_id;
  int64_t src_id = src.leader_op_id;
  auto optional_merged_id = TryMergeNode(cycle_detector.get(), dst_id, src_id);
  if (!optional_merged_id.has_value()) {
    // It forms a cycle.
    return;
  }
  if ((!dst.contract_dim_size == src.contract_dim_size)) {
    // Only support fuse gemms whose k dimension size are same.
    return;
  }
  dst.merge(src);
  dst.leader_op_id = *optional_merged_id;
}

class QuantizedDotShareOperandMergeConverter {
 public:
  QuantizedDotShareOperandMergeConverter(func::FuncOp func) : func_(func){};

  struct QuantizedDotInfoHash {
    std::size_t operator()(const Value& operand) const {
      std::size_t hash = mlir::hash_value(operand);
      return hash;
    }
  };

  void run();

  // record mapping of shared operand to custom operation group
  using ShareOperandMap =
      std::unordered_map<Value, SmallVector<mhlo_disc::CustomCallV2Op>,
                         QuantizedDotInfoHash>;
  void FillShareOperandMap(Block* block, ShareOperandMap& share_operand_map);
  void BuildQuantizedDotClusters(
      Block* block, ShareOperandMap& share_operand_map,
      SmallVector<QuantizedDotCluster>& merging_clusters);
  void applyMerging(QuantizedDotCluster& cluster);

 private:
  func::FuncOp func_;
};

void BuildBlockGraphCycles(Block* block,
                           std::unique_ptr<GraphCycles>& cycle_detector,
                           DenseMap<Operation*, int64_t>& op_to_node_id) {
  std::vector<Operation*> op_list;
  op_to_node_id.clear();
  for (Operation& op : *block) {
    op_to_node_id.try_emplace(&op, op_list.size());
    op_list.push_back(&op);
  }
  cycle_detector.reset(new GraphCycles(op_list.size()));
  for (int64_t node_id = 0; node_id < op_list.size(); node_id++) {
    Operation* op = op_list[node_id];
    for (Value operand : GetAllPossibleUsedValues(op)) {
      Operation* operand_op = operand.getDefiningOp();
      // Only consider the operand_op inside the target block.
      auto iter = op_to_node_id.find(operand_op);
      if (iter == op_to_node_id.end()) {
        continue;
      }
      cycle_detector->InsertEdge(iter->second, node_id);
    }
  }
}

void QuantizedDotShareOperandMergeConverter::FillShareOperandMap(
    Block* block, ShareOperandMap& share_operand_map) {
  block->walk([&](mhlo_disc::CustomCallV2Op op) {
    if (op.getCallTargetName() == "ral_pdll_qgemm") {
      // Only support lhs operand as shared operand. General situation will be
      // supported after meeting corresponding scenario and corresponding
      // quantized kernel is ready.
      Value lhs_operand = op.getOperands()[0];
      auto& operand_list = share_operand_map[std::move(lhs_operand)];
      operand_list.push_back(op);
    }
  });
}

void QuantizedDotShareOperandMergeConverter::BuildQuantizedDotClusters(
    Block* block, ShareOperandMap& share_operand_map,
    SmallVector<QuantizedDotCluster>& merging_clusters) {
  merging_clusters.clear();
  // Dot ops in `equal_merge_shape_map` can be merged together if the merging
  // does not introduce cycle.

  // Form cycle detector.
  std::unique_ptr<GraphCycles> cycle_detector(new GraphCycles(0));
  DenseMap<Operation*, int64_t> op_to_node_id;
  BuildBlockGraphCycles(block, cycle_detector, op_to_node_id);

  // Find merge clusters.
  for (auto& dots_can_merge : share_operand_map) {
    auto ops = dots_can_merge.second;
    if (ops.size() < 2) {
      continue;
    }
    SmallVector<QuantizedDotCluster> clusters;
    for (auto op : ops) {
      clusters.emplace_back(op.getOperation(),
                            op_to_node_id[op.getOperation()]);
    }
    for (int64_t i = 0; i < clusters.size(); i++) {
      auto& merged = clusters[i];
      if (merged.ops.empty()) {
        continue;
      }
      for (int64_t j = i + 1; j < clusters.size(); j++) {
        auto& to_merge = clusters[j];
        if (to_merge.ops.empty()) {
          continue;
        }
        // Try merge.
        TryMergeQuantizedDotClusters(merged, to_merge, cycle_detector);
      }
    }
    for (auto& cluster : clusters) {
      if (cluster.ops.size() > 1) {
        merging_clusters.push_back(cluster);
      }
    }
  }
}

void QuantizedDotShareOperandMergeConverter::applyMerging(
    QuantizedDotCluster& cluster) {
  auto& ops = cluster.ops;

  SmallVector<Value> concat_operands;
  int64_t concat_dim_sum = 0;
  for (auto op : ops) {
    mhlo_disc::CustomCallV2Op custom_qgemm_op =
        dyn_cast<mhlo_disc::CustomCallV2Op>(op);
    auto concat_operand = custom_qgemm_op.getOperands()[1];
    concat_operands.push_back(concat_operand);
    auto concat_op_type = concat_operand.getType().dyn_cast<RankedTensorType>();

    // qgemm layout is n x k
    auto concat_dim_size = concat_op_type.getDimSize(0);
    if (concat_dim_size == ShapedType::kDynamic) {
      concat_dim_sum = ShapedType::kDynamic;
    } else {
      concat_dim_sum += concat_dim_size;
    }
  }

  auto lead_op = ops.front();
  auto lead_custom_op = dyn_cast<mhlo_disc::CustomCallV2Op>(lead_op);

  auto lead_share_operand = lead_custom_op.getOperands()[0];
  auto lead_concat_operand = lead_custom_op.getOperands()[1];
  auto lead_concat_type =
      lead_concat_operand.getType().dyn_cast<RankedTensorType>();
  auto lead_operand_rank = lead_concat_type.getRank();

  // Only support weight concat whose rank is always 2 currently
  if (lead_operand_rank != 2) {
    return;
  }

  // delete dot ops in graph
  auto foremost = ops.front();
  for (int64_t i = 1; i < ops.size(); i++) {
    auto& op = ops[i];
    if (op->isBeforeInBlock(foremost)) {
      foremost = op;
    }
  }
  // Move all dot ops, and their consumers if necessary, before the original
  // foremost dot. This makes sure that the newly created ops in this function
  // dominates their uses.
  for (auto op : ops) {
    if (foremost == op) {
      continue;
    }
    op->moveBefore(foremost);
    ArrangeOperandsInsertPointInBlock(op);
  }

  int64_t lead_contract_dim = lead_concat_type.getDimSize(1);

  SmallVector<int64_t, 4> concat_op_shapes(2, ShapedType::kDynamic);

  concat_op_shapes[0] = concat_dim_sum;
  concat_op_shapes[1] = lead_contract_dim;

  // need check
  OpBuilder builder(lead_custom_op);

  auto loc = lead_op->getLoc();

  auto element_type = lead_concat_type.getElementType();
  auto concat_op_type = RankedTensorType::get(concat_op_shapes, element_type);
  // build concat op
  auto concat_op = builder.create<mhlo::ConcatenateOp>(
      loc, concat_op_type, concat_operands,
      builder.getI64IntegerAttr(0));  // concat dim is constrained.

  auto origin_result_type =
      lead_custom_op.getResults()[0].getType().dyn_cast<RankedTensorType>();
  auto result_rank = origin_result_type.getRank();
  SmallVector<int64_t, 4> result_shapes(result_rank, ShapedType::kDynamic);

  for (int i = 0; i < result_rank; i += 1) {
    if (i == result_rank - 1) {
      result_shapes[i] = concat_dim_sum;
    } else {
      result_shapes[i] = origin_result_type.getDimSize(i);
    }
  }
  auto result_type = RankedTensorType::get(result_shapes, element_type);

  // Broadcast scale and zeropoint to the same size with bias(channel number)
  // For instance:
  // scale: [1.0] broadcast to scale_vector: [1.0, 1.0 ... 1.0],
  // len(scale_vector) == len(bias)
  SmallVector<SmallVector<Value>> scale_operands(7);
  for (auto op : ops) {
    auto op_operands = op->getOperands();
    auto weight_operand = op_operands[1];
    // layout of weight is nxk
    int64_t broadcast_dim_size =
        weight_operand.getType().dyn_cast<RankedTensorType>().getDimSize(0);
    scale_operands[0].push_back(op_operands[2]);
    for (int i = 1; i < 7; i += 1) {
      auto to_bcast_operand = op_operands[2 + i];
      auto bcast_elem_type = to_bcast_operand.getType()
                                 .dyn_cast<RankedTensorType>()
                                 .getElementType();
      SmallVector<int64_t> broadcast_op_shapes = {broadcast_dim_size};
      auto bcast_type =
          RankedTensorType::get(broadcast_op_shapes, bcast_elem_type);
      auto bcast_op = builder.create<mhlo::BroadcastInDimOp>(
          loc, bcast_type, to_bcast_operand, builder.getI64TensorAttr({}));
      scale_operands[i].push_back(bcast_op);
    }
  }

  // Concat scale vector and zeropoint vector in same shared operand group
  // len(scale_after_concat) == concat_dim_sum
  SmallVector<Value> scale_concat_operands;
  for (int i = 0; i < 7; i += 1) {
    int64_t concat_dim_sum = 0;
    for (auto op : scale_operands[i]) {
      int64_t concat_dim_size =
          op.getType().dyn_cast<RankedTensorType>().getDimSize(0);
      concat_dim_sum += concat_dim_size;
    }

    SmallVector<int64_t, 4> concat_op_shapes = {concat_dim_sum};
    auto element_type = scale_operands[i][0]
                            .getType()
                            .dyn_cast<RankedTensorType>()
                            .getElementType();
    auto concat_op_type = RankedTensorType::get(concat_op_shapes, element_type);
    auto concat_op = builder.create<mhlo::ConcatenateOp>(
        loc, concat_op_type, scale_operands[i],
        builder.getI64IntegerAttr(0));  // concat dim is constrained.
    scale_concat_operands.push_back(concat_op);
  }

  SmallVector<Value> new_operands = {lead_share_operand, concat_op};

  for (auto scale_concat_operand : scale_concat_operands) {
    new_operands.push_back(scale_concat_operand);
  }

  SmallVector<Type> newResultTypes;
  newResultTypes.push_back(result_type);

  auto newCustomOp = builder.create<mhlo_disc::CustomCallV2Op>(
      loc, newResultTypes, new_operands, "ral_pdll_qgemm_per_channel",
      lead_custom_op.getCustomAttrs(), lead_custom_op.getHasSideEffect(),
      lead_custom_op.getDevice(), lead_custom_op.getInputPlacements(),
      lead_custom_op.getOutputPlacements(),
      lead_custom_op.getExpectedInputLayouts(),
      lead_custom_op.getExpectedOutputLayouts(),
      lead_custom_op.getExpectedInputLayouts(),
      lead_custom_op.getExpectedOutputLayouts());

  auto newResults = newCustomOp->getResults();

  int concat_dim_in_result = result_rank - 1;

  if (result_type.getNumDynamicDims() == 0) {
    int64_t concat_dim_start = 0;
    for (int64_t i = 0; i < ops.size(); i += 1) {
      mhlo_disc::CustomCallV2Op op =
          dyn_cast<mhlo_disc::CustomCallV2Op>(ops[i]);
      auto orig_customop_type =
          op.getResults()[0].getType().dyn_cast<RankedTensorType>();

      SmallVector<int64_t> start(result_rank, 0);
      SmallVector<int64_t> limit(result_rank);
      SmallVector<int64_t> strides(result_rank, 1);
      start[concat_dim_in_result] = concat_dim_start;
      for (int j = 0; j < result_rank; j += 1) {
        if (j == concat_dim_in_result) {
          limit[j] = concat_dim_start +
                     orig_customop_type.getDimSize(concat_dim_in_result);
          concat_dim_start = limit[j];
        } else {
          limit[j] = orig_customop_type.getDimSize(j);
        }
      }
      Value slice = builder.create<mhlo::SliceOp>(
          loc, newResults[0], GetI64ElementsAttr(start, &builder),
          GetI64ElementsAttr(limit, &builder),
          GetI64ElementsAttr(strides, &builder));

      auto cast_slice = builder.create<tensor::CastOp>(
          loc, op->getResult(0).getType(), slice);

      op->replaceAllUsesWith(cast_slice);
    }
  } else {
    // Use dynamic-dim ops.
    Value concat_dim_start = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);

    for (int64_t i = 0; i < ops.size(); i += 1) {
      mhlo_disc::CustomCallV2Op op =
          dyn_cast<mhlo_disc::CustomCallV2Op>(ops[i]);
      auto concatenate_op = op.getOperands()[1];
      auto non_concatenate_op = op.getOperands()[0];
      auto orig_customop_type =
          op.getResults()[0].getType().dyn_cast<RankedTensorType>();

      SmallVector<Value, 4> start_values(result_rank, zero);
      SmallVector<Value, 4> limit_values(result_rank);
      SmallVector<Value, 4> strides_values(result_rank, one);

      for (int64_t j = 0; j < result_rank; j += 1) {
        if (j == concat_dim_in_result) {
          start_values[j] = concat_dim_start;
          limit_values[j] = builder.create<arith::AddIOp>(
              loc, start_values[j],
              builder.create<tensor::DimOp>(loc, concatenate_op,
                                            concat_dim_in_result - 1));
          concat_dim_start = limit_values[j];
        } else {
          limit_values[j] =
              builder.create<tensor::DimOp>(loc, non_concatenate_op, j);
        }
      }

      auto index_ty = builder.getIndexType();
      auto start_indices = builder.create<tensor::FromElementsOp>(
          loc,
          RankedTensorType::get({static_cast<int64_t>(start_values.size())},
                                index_ty),
          start_values);
      auto limit_indices = builder.create<tensor::FromElementsOp>(
          loc,
          RankedTensorType::get({static_cast<int64_t>(limit_values.size())},
                                index_ty),
          limit_values);
      auto strides_indices = builder.create<tensor::FromElementsOp>(
          loc,
          RankedTensorType::get({static_cast<int64_t>(strides_values.size())},
                                index_ty),
          strides_values);
      SmallVector<int64_t, 4> slice_shapes(result_rank, ShapedType::kDynamic);
      for (int64_t j = 0; j < result_rank; j++) {
        slice_shapes[j] = orig_customop_type.getDimSize(j);
      }
      auto slice_type = RankedTensorType::get(slice_shapes, element_type);

      Value dyn_slice = builder.create<mhlo::RealDynamicSliceOp>(
          loc, slice_type, newResults[0], start_indices, limit_indices,
          strides_indices);

      auto cast_dyn_slice = builder.create<tensor::CastOp>(
          loc, op->getResult(0).getType(), dyn_slice);

      op->replaceAllUsesWith(cast_dyn_slice);
    }
  }
  // No longer need the original quantized dot ops.
  for (int64_t i = 0; i < ops.size(); i++) {
    ops[i]->erase();
  }
}

void QuantizedDotShareOperandMergeConverter::run() {
  SmallVector<Block*> blocks;
  func_.walk([&](Block* block) { blocks.push_back(block); });
  for (Block* block : blocks) {
    ShareOperandMap share_input_map;
    FillShareOperandMap(block, share_input_map);

    SmallVector<QuantizedDotCluster> merging_clusters;
    BuildQuantizedDotClusters(block, share_input_map, merging_clusters);

    for (auto& cluster : merging_clusters) {
      applyMerging(cluster);
    }
  }
}

struct DiscQuantizedDotMergePass
    : public DiscQuantizedDotMergePassBase<DiscQuantizedDotMergePass> {
  DiscQuantizedDotMergePass()
      : DiscQuantizedDotMergePassBase<
            DiscQuantizedDotMergePass>::DiscQuantizedDotMergePassBase() {}
  void runOnOperation() override;

 private:
  void quantizedDotShareOperandMerging(func::FuncOp& func);
};

void DiscQuantizedDotMergePass::runOnOperation() {
  func::FuncOp func = getOperation();
  quantizedDotShareOperandMerging(func);
}

void DiscQuantizedDotMergePass::quantizedDotShareOperandMerging(
    func::FuncOp& func) {
  QuantizedDotShareOperandMergeConverter(func).run();
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createDiscQuantizedDotMergePass() {
  return std::make_unique<DiscQuantizedDotMergePass>();
}

}  // namespace disc_ral
}  // namespace mlir

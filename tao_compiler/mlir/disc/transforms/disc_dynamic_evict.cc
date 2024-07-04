// Copyright 2021 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file implements logic for lowering HLO DISC dialect to LHLO DISC
// dialect.

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <list>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "lhlo/IR/lhlo_ops.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/disc/IR/disc_shape_ops.h"
#include "mlir/disc/IR/lhlo_disc_ops.h"
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/fusion_utils.h"
#include "mlir/disc/transforms/placement_utils.h"
#include "mlir/disc/transforms/rewriters.h"
#include "mlir/disc/transforms/shape_utils.h"
#include "tensorflow/tsl/platform/default/logging.h"

namespace mlir {
using placement_utils::kDiscPlaceAssignment;
using placement_utils::kGpu;

namespace mhlo_disc {
namespace {

bool IsRematerializable(const Operation* op) { return true; }

enum class RematStrategy {
  // Recompute the node at a later program point.
  kRecompute,
  // Change the layout into a compact form and uncompress it back at a later
  // program point.
  kCompress,
  // Copy the data off the device to the host to be copied back later.
  kHostOffload,

  // Combination of different strategies.
  kRecomputeAndCompress,
  kRecomputeAndHostOffload,
  kCompressAndHostOffload,
  kAll,
  kNoAction,
};

struct CompactShape {};
struct Item {
  Value memref;
  std::vector<int> live_range;
  std::vector<int> compacted_live_range;
  bool live_out;
  bool inplace_reuse;
  bool is_persistent;
  bool info_outdated;
};

class HistoryItems {
 public:
  HistoryItems() = default;

  Item ConstructItemFromValue(
      const Value memref, std::unordered_map<int64_t, int>& op_position_map) {
    // Add memrefs and their live range.
    std::vector<int> live_range;
    std::vector<int> compacted_live_range;

    live_range.push_back(
        op_position_map[reinterpret_cast<int64_t>(memref.getDefiningOp())]);
    compacted_live_range.push_back(
        op_position_map[reinterpret_cast<int64_t>(memref.getDefiningOp())]);

    bool live_out = false, inplace_reuse = false;
    for (auto user : memref.getUsers()) {
      Operation* fusion_parent = nullptr;
      if (auto parent_op = user->getParentOfType<lmhlo::FusionOp>()) {
        fusion_parent = parent_op;
      }
      if (isa<func::ReturnOp>(user)) {
        live_out = true;
        continue;
      }

      if (isa<lmhlo_disc::ArgsMutationOp>(user) &&
          user->getOperand(0) == memref) {
        inplace_reuse = true;
        continue;
      }

      live_range.push_back(op_position_map[reinterpret_cast<int64_t>(user)]);
      if (fusion_parent) {
        compacted_live_range.push_back(
            op_position_map[reinterpret_cast<int64_t>(fusion_parent)]);
      } else {
        compacted_live_range.push_back(
            op_position_map[reinterpret_cast<int64_t>(user)]);
      }
    }

    // Sort and remove duplicates.
    // Duplicates happen inside a fusion block
    std::sort(compacted_live_range.begin(), compacted_live_range.end());
    compacted_live_range.erase(
        std::unique(compacted_live_range.begin(), compacted_live_range.end()),
        compacted_live_range.end());

    std::sort(live_range.begin(), live_range.end());

    return Item{memref,
                live_range,
                compacted_live_range,
                live_out,
                inplace_reuse,
                false,
                false};
  }

  void Add(const Value memref,
           std::unordered_map<int64_t, int>& op_position_map) {
    // Add memrefs and their live range.
    auto item = this->ConstructItemFromValue(memref, op_position_map);
    Add(item);
  }

  void Add(const Item& item) {
    int64_t key = reinterpret_cast<int64_t>(item.memref.getAsOpaquePointer());
    live_range_map_[key] = history_items_.size();
    history_items_.push_back(item);
  }

  bool IsExist(const Value memref) {
    int64_t key = reinterpret_cast<int64_t>(memref.getAsOpaquePointer());
    return live_range_map_.find(key) != live_range_map_.end();
  }

  bool LiveAcross(const Value memref, int position) {
    auto& item = GetItem(memref);
    return item.is_persistent || item.live_out ||
           item.live_range.back() > position;
  }

  bool LiveAcross(const Item& item, int position) {
    return item.is_persistent || item.live_out ||
           item.live_range.back() > position;
  }

  void Update(Value memref, Item& target_item) {
    auto index =
        live_range_map_[reinterpret_cast<int64_t>(memref.getAsOpaquePointer())];
    history_items_[index] = target_item;
  }

  Item& GetItem(Value memref) {
    auto index =
        live_range_map_[reinterpret_cast<int64_t>(memref.getAsOpaquePointer())];
    return history_items_[index];
  }

  std::vector<Item>& GetHistoryItems() { return history_items_; }

  void ResetStatus() {
    history_items_.clear();
    live_range_map_.clear();
  }

 private:
  std::vector<Item> history_items_;
  std::map<int64_t, int> live_range_map_;
};

class MemoryUsageTracker {
 public:
  struct RematEvalResult {
    Value original_buffer;
    Value recomputed_buffer;
    double score;
    RematStrategy strategy;
    std::vector<Operation*> remat_block;

    // Infos needed to pass to runtime
    double normalized_recomputation_cost;
    double normalized_memory_saving;
    double distance_to_next_usage;
  };

  struct RematStats {
    int recomputed_item_count;
    int offloaded_item_count;
    int compressed_item_count;
  };

  MemoryUsageTracker() = default;

  void SetRematStrategy(RematStrategy strategy) { remat_strategy_ = strategy; }

  //void SetSymblicDimMgr(SymbolicDimMgr& symbolic_dim_mgr) {
  //  symbolic_dim_mgr_ = symbolic_dim_mgr;
  //}
  void SetAllOperationPositionInfo(
      const std::unordered_map<int64_t, int>& operation_position_map,
      const std::unordered_map<int, int64_t>& reverse_operation_position_map) {
    operation_position_map_ = operation_position_map;
    reverse_operation_position_map_ = reverse_operation_position_map;
  }

  void ProcessAlloc(memref::AllocOp op) {
    auto memref = op.getResult();
    auto item =
        history_items_.ConstructItemFromValue(memref, operation_position_map_);
    if (NeedSkip(item)) {
      return;
    }
    history_items_.Add(item);
  }

  void ProcessDealloc(memref::DeallocOp op) {
    auto memref = op.getOperand();
    if (!history_items_.IsExist(memref)) {
      return;
    }
  }

  void ProcessCustomCallV2(lmhlo_disc::CustomCallV2Op op) {
    for (auto memref : op.getResults()) {
      auto item =
          history_items_.ConstructItemFromValue(memref, operation_position_map_);
      history_items_.Add(item);
    }
  }

  void ProcessConstant(lmhlo::ConstantOp op) {
    auto memref = op.getOperand();
    auto item =
        history_items_.ConstructItemFromValue(memref, operation_position_map_);
    item.is_persistent = true;
    history_items_.Add(item);
  }

  void ProcessArgument(Value memref) {
    auto item =
        history_items_.ConstructItemFromValue(memref, operation_position_map_);
    item.is_persistent = true;
    history_items_.Add(item);
  }

  void ProcessRematOp(lmhlo_disc::EvictOp op) {
    for (auto memref : op.getResults()) {
      auto item =
          history_items_.ConstructItemFromValue(memref, operation_position_map_);
      history_items_.Add(item);
    }
  }

  void ProcessRematOp(lmhlo_disc::EvictionCheckOp op) {
    for (auto memref : op.getResults()) {
      auto item =
          history_items_.ConstructItemFromValue(memref, operation_position_map_);
      history_items_.Add(item);
    }
  }

  void ProcessScfIfOp(scf::IfOp op) {
    for (auto memref : op.getResults()) {
      auto item =
          history_items_.ConstructItemFromValue(memref, operation_position_map_);
      history_items_.Add(item);
    }
  }

  bool IsStaticShape(Value memref) {
    auto memref_ty = memref.getType().dyn_cast_or_null<MemRefType>();
    if (!memref_ty) {
      return false;
    }
    assert(memref_ty.getLayout().isIdentity());
    return memref_ty.hasStaticShape();
  }

  size_t GetMemoryUsageForValue(Value memref) {
    auto memref_ty = memref.getType().dyn_cast_or_null<MemRefType>();
    if (!memref_ty) {
      return 0;
    }
    assert(memref_ty.getLayout().isIdentity());
    if (memref_ty.hasStaticShape()) {
      int byte_width = memref_ty.getElementTypeBitWidth() / 8;
      auto shape = memref_ty.getShape();

      size_t logical_size = byte_width;
      for (size_t dimSize : shape) {
        logical_size *= dimSize;
      }
      return logical_size;
    } else {
      // not implemented
      throw std::logic_error("Size for dynamic shape not implemented");
    }
  }

  CompactShape GetCompactShape(Value memref) {
    throw std::logic_error("GetCompactShape not implemented");
  }
  size_t GetCompactedMemoryUsageForItem(const Item& item) {
    throw std::logic_error("GetCompactedMemoryUsageForItem not implemented");
  }

  void InsertRecomputationSubgraph(/*Operation* op, OpBuilder& rewriter, */Block* block, RematEvalResult& eval_res) {

    /*******0. Insert recomputation subgraph of result.item.memref*********/

    for(auto op_idx = 0; op_idx < eval_res.remat_block.size(); op_idx++) {
      eval_res.remat_block[op_idx]->moveBefore(block, block->end());
      //eval_res.remat_block[op_idx]->replaceUsesOfWith(eval_res.original_buffer, eval_res.recomputed_buffer);
    }

    //auto output_buffer = eval_res.remat_block[eval_res.remat_block.size() - 2]->getResult(0);
    //auto terminator_op = rewriter.create<scf::YieldOp>(op->getLoc(), /*operands=*/output_buffer);
    //terminator_op.getOperation()->moveBefore(block, block->end());

    /*******1. Update Status*********/
    remat_stats_.recomputed_item_count++;
  }

  void Offload(OpBuilder& builder, int op_position, RematEvalResult& result) {
    throw std::logic_error("Offload not implemented");
  }

  void Compress(OpBuilder& builder, int op_position, RematEvalResult& result) {
    throw std::logic_error("Compress not implemented");
  }

  RematEvalResult EvalOffload(OpBuilder& rewriter, int op_position, const Item& item,
                              size_t target_peak_memory = -1) {
    RematEvalResult eval_res{
        item.memref, item.memref, (double)kInvalidScore, RematStrategy::kRecompute, {}, 0, 0, 0};
    return eval_res;
  }

  RematEvalResult EvalCompression(OpBuilder& rewriter, int op_position, const Item& item,
                                  size_t target_peak_memory = -1) {
    RematEvalResult eval_res{
        item.memref, item.memref, (double)kInvalidScore, RematStrategy::kRecompute, {}, 0, 0, 0};
    return eval_res;
  }

  bool RecursiveSearch(OpBuilder& rewriter, Operation* parent_op, int operand_idx, Value input_buffer, RematEvalResult& eval_res, int trigger_point_position, bool create_op, int& current_recompute_graph_size) {
    auto is_buffer_live_across = [&](Value buffer, int trigger_point_position) {
      auto& buffer_item = history_items_.GetItem(buffer);
      return history_items_.LiveAcross(buffer_item, trigger_point_position);
    };

    auto is_input_lifetime_extended = [&is_buffer_live_across, this](Operation* leaf_op, int trigger_point_position) {
      if (isa<lmhlo_disc::CustomCallV2Op>(leaf_op)) {
        for (auto input : leaf_op->getOperands()) {
          if (!is_buffer_live_across(input, trigger_point_position)) {
            return true;
          }
        }
      }

      int num_input_operand = leaf_op->getNumOperands() -
                              disc_ral::getNumResultOperands(leaf_op);
      for (auto idx = 0; idx < num_input_operand; ++idx) {
        if (!is_buffer_live_across(leaf_op->getOperand(idx), trigger_point_position)) {
          return true;
        }
      }

      return false;
    };

    auto is_memory_expanding = [&](Operation* leaf_op) {
      if(isa<lmhlo::BroadcastInDimOp, lmhlo::DynamicBroadcastInDimOp>(leaf_op)) return true;
      if(isa<lmhlo::ConvertOp>(leaf_op)) {
        auto in = leaf_op->getOperand(0);
        auto out = leaf_op->getOperand(1);
        auto in_type = in.getType().dyn_cast<MemRefType>().getElementType();
        auto out_type = out.getType().dyn_cast<MemRefType>().getElementType();
        return (in_type.isF16() || in_type.isBF16()) &&
              (out_type.isF32() || out_type.isF64());
      }
      return false;
    };
    auto is_search_ending = [&is_memory_expanding, &is_input_lifetime_extended, this] (Operation* op, int trigger_point_position) {
      return is_memory_expanding(op) || !is_input_lifetime_extended(op, trigger_point_position);
    };

    auto is_remat_block = [&](Operation* op) {
      return (isa<scf::IfOp>(op) && op->hasAttr("remat_block"));
    };

    auto leaf_op = GetBufferDefiningOp(input_buffer);

    // For now, we just return false
    if(leaf_op == nullptr) return false;

    // Search through remat block
    if(is_remat_block(leaf_op)) {
      llvm::dbgs() << "Search through remat block \n";
      auto memref_load_op = leaf_op->getOperand(0).getDefiningOp<memref::LoadOp>();
      auto eviction_check_op = memref_load_op->getOperand(0).getDefiningOp<lmhlo_disc::EvictionCheckOp>();
      leaf_op = eviction_check_op->getOperand(0).getDefiningOp<lmhlo_disc::EvictOp>();
      input_buffer = eviction_check_op->getOperand(0);
    }

    // If is EvictOp, we should search through
    if(isa<lmhlo_disc::EvictOp>(leaf_op)) {
      for(auto idx = 0; idx < leaf_op->getNumResults(); idx ++) {
        if(input_buffer == leaf_op->getResult(idx)) {
          return RecursiveSearch(rewriter, parent_op, operand_idx, leaf_op->getOperand(idx), eval_res, trigger_point_position, create_op, current_recompute_graph_size);
        }
      }
    }

    // Update remat block
    auto alloc_op = input_buffer.getDefiningOp();

    if(alloc_op == nullptr) return false;

    current_recompute_graph_size += 1;
    if(current_recompute_graph_size > kMaxRecomputeBlockSize) return false;

    auto cloned_compute_op = leaf_op;
    if(create_op && isa<memref::AllocOp>(alloc_op)) {
      auto cloned_alloc_op = rewriter.clone(*alloc_op);
      cloned_compute_op = rewriter.clone(*leaf_op);
      cloned_compute_op->replaceUsesOfWith(input_buffer, cloned_alloc_op->getResult(0));
 
      parent_op->replaceUsesOfWith(parent_op->getOperand(operand_idx), cloned_alloc_op->getResult(0));
      
      eval_res.remat_block.insert(eval_res.remat_block.begin(), cloned_compute_op);
      eval_res.remat_block.insert(eval_res.remat_block.begin(), cloned_alloc_op);
    } else if(create_op && isa<lmhlo_disc::CustomCallV2Op>(alloc_op)) {
      auto cloned_alloc_op = rewriter.clone(*alloc_op);
      parent_op->replaceUsesOfWith(parent_op->getOperand(operand_idx), cloned_alloc_op->getResult(0));
      eval_res.remat_block.insert(eval_res.remat_block.begin(), cloned_compute_op);
    } else {
      return false;
    }

    if(is_search_ending(leaf_op, trigger_point_position)) {
      return true;
    }

    if((isa<lmhlo_disc::CustomCallV2Op>(leaf_op) && leaf_op->getNumResults() != 1)
      || (isa<scf::IfOp>(leaf_op) && !is_remat_block(leaf_op))
      || (isa<lmhlo::LmhloOp>(leaf_op) && disc_ral::getNumResultOperands(leaf_op) != 1)) return false;

    if(isa<lmhlo_disc::CustomCallV2Op>(leaf_op)) {
      int idx = 0;
      for (auto input : leaf_op->getOperands()) {
        if (input != input_buffer && !is_buffer_live_across(input, trigger_point_position)) {
          // Get defininig op for input and continue search
          if(GetBufferDefiningOp(input)) {
            return RecursiveSearch(rewriter, cloned_compute_op, idx, input, eval_res, trigger_point_position, create_op, current_recompute_graph_size);
          } else {
            return false;
          }
        }
        idx += 1;
      }
    }

    if(isa<lmhlo::LmhloOp>(leaf_op)) {
      int num_input_operand = leaf_op->getNumOperands() -
                              disc_ral::getNumResultOperands(leaf_op);
      for (auto idx = 0; idx < num_input_operand; ++idx) {
        auto input = leaf_op->getOperand(idx);
        if (input != input_buffer && !is_buffer_live_across(input, trigger_point_position)) {
          // Get defining op for input and continue search
          if(GetBufferDefiningOp(input)) {
            return RecursiveSearch(rewriter, cloned_compute_op, idx, input, eval_res, trigger_point_position, create_op, current_recompute_graph_size);
          } else {
            return false;
          }
        }
      }
    }

    // Should not reach here
    return false;    
  }

  Operation* GetBufferDefiningOp(Value buffer_to_check) {
    auto is_write_op = [&](Operation* op, Value memref) {
      if (isa<lmhlo_disc::CustomCallV2Op, lmhlo_disc::EvictOp, lmhlo_disc::EvictionCheckOp, scf::IfOp>(op)) {
        for (auto result : op->getResults()) {
          if (memref == result) return true;
        }
        return false;
      }
      if(isa<lmhlo::LmhloOp>(op)) {
        int num_input_operand =
          op->getNumOperands() - disc_ral::getNumResultOperands(op);
        for (auto idx = num_input_operand; idx < op->getNumOperands(); ++idx) {
          if (memref == op->getOperand(idx)) {
            return true;
          }
        }
      }
      return false;
    };

    auto get_buffer_definig_op = [&is_write_op, this](Value buffer) -> Operation* {
      auto item = history_items_.GetItem(buffer);
      for (int idx = 0; idx < item.live_range.size(); ++idx) {
        auto user = reinterpret_cast<Operation*>(
            reverse_operation_position_map_[item.live_range[idx]]);
        if (is_write_op(user, item.memref)) {
          return user;
        }
      }
      return nullptr;
    };

    return get_buffer_definig_op(buffer_to_check);
  }


  void SearchAndCreateRecomputationSubgraph(OpBuilder& rewriter, Value buffer, RematEvalResult& eval_res, int trigger_point_position) {
    
    Value current_buffer = buffer;
    Operation* defining_op = nullptr;

    auto is_remat_block = [&](Operation* op) {
      return isa<scf::IfOp>(op) && op->hasAttr("remat_block");
    };

    while (1) {
      defining_op = GetBufferDefiningOp(current_buffer);

      // Search through remat_block
      if(is_remat_block(defining_op)) {
        auto memref_load_op = defining_op->getOperand(0).getDefiningOp<memref::LoadOp>();
        auto eviction_check_op = memref_load_op->getOperand(0).getDefiningOp<lmhlo_disc::EvictionCheckOp>();
        defining_op = eviction_check_op->getOperand(0).getDefiningOp<lmhlo_disc::EvictOp>();
        current_buffer = eviction_check_op->getOperand(0);
      }

      if(!isa<lmhlo_disc::EvictOp>(defining_op)) {
        eval_res.original_buffer = current_buffer;
        break;
      }
      for(auto idx = 0; idx < defining_op->getNumResults(); idx ++) {
        if(current_buffer == defining_op->getResult(idx)) {
          current_buffer = defining_op->getOperand(idx);
          break;
        }
      }
    }
    
    SearchAndCreateRecomputationSubgraph(rewriter, defining_op, eval_res, trigger_point_position, true);
  }


  void SearchAndCreateRecomputationSubgraph(OpBuilder& rewriter, Operation* defining_op, RematEvalResult& eval_res, int trigger_point_position, bool create_op) {
      auto cloned_computation_op = defining_op;
      if(create_op) {
        cloned_computation_op = rewriter.clone(*defining_op);
        auto alloc_op = eval_res.original_buffer.getDefiningOp<memref::AllocOp>();
        auto cloned_alloc_op = rewriter.clone(*alloc_op);
        //auto realloc_op = rewriter.create<lmhlo_disc::ReallocOp>(defining_op->getLoc(), result_type, eval_res.recomputed_buffer);
        cloned_computation_op->replaceUsesOfWith(eval_res.original_buffer, cloned_alloc_op->getResult(0));
        eval_res.recomputed_buffer = cloned_alloc_op->getResult(0);
        // alloc first, then compute, but in reverse order
        eval_res.remat_block.insert(eval_res.remat_block.begin(), cloned_computation_op);
        eval_res.remat_block.insert(eval_res.remat_block.begin(), cloned_alloc_op);
      }

      int current_recompute_graph_size = 1;
      bool search_success = true;

      auto is_buffer_live_across = [&](Value buffer, int trigger_point_position) {
        auto& buffer_item = history_items_.GetItem(buffer);
        return history_items_.LiveAcross(buffer_item, trigger_point_position);
      };

      int idx = 0;
      for(auto input : defining_op->getOperands()) {
        if(input != eval_res.original_buffer && !is_buffer_live_across(input, trigger_point_position)) {
          if(!RecursiveSearch(rewriter, cloned_computation_op, idx, input, eval_res, trigger_point_position, create_op, current_recompute_graph_size)) {
            search_success = false;
            break;
          }
        }
        idx += 1;
      }
      
      if(search_success) {
        eval_res.score = kDefaultValidScore;
      }

    return;
  }

  RematEvalResult EvalRecomputation(OpBuilder& rewriter, int trigger_point_position, const Item& item,
                                         size_t target_peak_memory = -1) {
    
    RematEvalResult eval_res{
        item.memref, item.memref, (double)kInvalidScore, RematStrategy::kRecompute, {}, 0, 0, 0};

    // Filter out buffers that are too small
    if (IsStaticShape(item.memref) && GetMemoryUsageForValue(item.memref) < kSmallMemrefSize) {
      return eval_res;
    }

    // Find the defining op, definig op is the bottom root op of recomputation subgraph for this item.memref
    // In SSA format, defining_op is the op that write valid data into this memref
    Operation* defining_op = nullptr;
    auto is_write_op = [&](Operation* op, Value memref) {
      if (isa<lmhlo_disc::CustomCallV2Op, lmhlo_disc::EvictOp, lmhlo_disc::EvictionCheckOp, scf::IfOp>(op)) {
        for (auto result : op->getResults()) {
          if (memref == result) return true;
        }
        return false;
      }
      if(isa<lmhlo::LmhloOp>(op)) {
        int num_input_operand =
          op->getNumOperands() - disc_ral::getNumResultOperands(op);
        for (auto idx = num_input_operand; idx < op->getNumOperands(); ++idx) {
          if (memref == op->getOperand(idx)) {
            return true;
          }
        }
      }
      return false;
    };

    auto is_comm_op = [&](Operation* op) {
      if (!isa<lmhlo_disc::CustomCallV2Op>(op)) {
        return false;
      }

      std::string call_target_name =
          op->getAttr("call_target_name").cast<StringAttr>().getValue().str();

      return (call_target_name == "ral_all_gather" ||
              call_target_name == "ral_all_reduce" ||
              call_target_name == "ral_all_to_all" ||
              call_target_name == "ral_reduce_scatter");
    };

    auto is_remat_block = [&](Operation* op) {
      return isa<scf::IfOp>(op) && op->hasAttr("remat_block");
    };

    auto is_customcall_op = [&](Operation* op) {
      return isa<lmhlo_disc::CustomCallV2Op>(op);
    };

    // Comm ops are not recomputable.
    // If a buffer is already recomputed, we dont recompute it again.
    auto is_unrecomputable_op = [&is_comm_op, &is_remat_block, &is_customcall_op](Operation* op) {
      return is_comm_op(op) || is_customcall_op(op) || (isa<lmhlo::LmhloOp>(op) && disc_ral::getNumResultOperands(op) != 1);
    };

    int start_position = -1, end_position = -1;
    for (int idx = 0; idx < item.live_range.size(); ++idx) {
      auto user = reinterpret_cast<Operation*>(
          reverse_operation_position_map_[item.live_range[idx]]);
      if (is_write_op(user, item.memref)) {
        // We only process SSA format IR
        if (defining_op != nullptr) {
          return eval_res;
        }
        // We dont recompute communication op now, cause it might cause deadlock
        // across different ranks, but this buffer can be offloaded.
        if (is_unrecomputable_op(user)) {
          return eval_res;
        }
        defining_op = user;
      }

      if (item.live_range[idx] > trigger_point_position) {
        // idx=0 is the defining op which create this memref
        // We dont recompute buffer that are just allocated
        if (idx == 1) {
          return eval_res;
        }
        end_position = item.live_range[idx];
        start_position = item.live_range[idx - 1];
        break;
      }
    }

    if(start_position >= trigger_point_position) {
      return eval_res;
    }

    // We dont want the interval between current position and next time we use
    // this memref to be too small When we count the interval, we exclude
    // alloc/dealloc/constant/argsmutation and ops inside the same fusion block
    Operation* last_fusion_parent = nullptr;
    int temp_position = trigger_point_position;
    int interval = end_position - trigger_point_position;
    int original_interval = interval;

    if (auto parent_op = reinterpret_cast<Operation*>(
                             reverse_operation_position_map_[temp_position])
                             ->getParentOfType<lmhlo::FusionOp>()) {
      last_fusion_parent = parent_op;
    }
    while (temp_position < end_position) {
      if (auto parent_op = reinterpret_cast<Operation*>(
                               reverse_operation_position_map_[temp_position])
                               ->getParentOfType<lmhlo::FusionOp>()) {
        interval = (last_fusion_parent == parent_op) ? interval - 1 : interval;
        last_fusion_parent = parent_op;
      }

      auto tmp_op = reinterpret_cast<Operation*>(reverse_operation_position_map_[temp_position]);
      if (isa<lmhlo_disc::ArgsMutationOp, lmhlo::ConstantOp, memref::AllocOp,
              memref::AllocaOp, memref::DeallocOp, scf::IfOp>(tmp_op)) {
        interval -= 1;
      } else if(tmp_op->getName().getDialectNamespace() == "arith" || tmp_op->getName().getDialectNamespace() == "memref") {
        interval -= 1;
      }
      temp_position += 1;
    }
    if (interval < kMinimumInterval) {
      return eval_res;
    }

    // A already EvictOp result could be Evict again and we already know it has a valid recomputation subgraph
    if(isa<lmhlo_disc::EvictOp>(defining_op) || is_remat_block(defining_op)) {
      if(is_remat_block(defining_op)) {
        llvm::dbgs() << "A remat block is selected as recompute candidate again: " << item.memref << "\n";
      }
      eval_res.score = kDefaultValidScore;
      return eval_res;
    }

    SearchAndCreateRecomputationSubgraph(rewriter, defining_op, eval_res, trigger_point_position, false);
    return eval_res;
  }

  RematEvalResult GetRematEvaluation(OpBuilder& rewriter,int op_position, const Item& item) {
    switch (remat_strategy_) {
      case RematStrategy::kRecompute:
        return EvalRecomputation(rewriter, op_position, item);
      case RematStrategy::kCompress:
        return EvalCompression(rewriter, op_position, item);
      case RematStrategy::kHostOffload:
        return EvalOffload(rewriter, op_position, item);
      case RematStrategy::kRecomputeAndHostOffload: {
        auto recompute_eval = EvalRecomputation(rewriter, op_position, item);
        auto offload_eval = EvalOffload(rewriter, op_position, item);
        if (recompute_eval.score == kInvalidScore) return offload_eval;
        return recompute_eval;
      }
      case RematStrategy::kRecomputeAndCompress: {
        auto recompute_eval = EvalRecomputation(rewriter, op_position, item);
        auto compress_eval = EvalCompression(rewriter, op_position, item);
        if (recompute_eval.score >= compress_eval.score) return recompute_eval;
        return compress_eval;
      }
      case RematStrategy::kAll: {
        auto recompute_eval = EvalRecomputation(rewriter, op_position, item);
        auto compress_eval = EvalCompression(rewriter, op_position, item);
        auto offload_eval = EvalOffload(rewriter, op_position, item);
        if (recompute_eval.score >= compress_eval.score &&
            recompute_eval.score >= offload_eval.score)
          return recompute_eval;
        if (compress_eval.score >= recompute_eval.score &&
            compress_eval.score >= offload_eval.score)
          return compress_eval;
        return offload_eval;
      }
      default:
        return EvalRecomputation(rewriter, op_position, item);
    }
  }

  bool EvictLivingCandidates(OpBuilder& rewriter, std::vector<RematEvalResult>& candidate_evals, Operation* compute_op, int64_t trigger_point_position) {
  
    if(candidate_evals.size() == 0) return false;
    // Collect candidate buffers
    SmallVector<Value> candidate_buffers;
    for(auto candidate_eval : candidate_evals) {
      candidate_buffers.push_back(candidate_eval.original_buffer);
    }

    SmallVector<mlir::Type> result_types;
    for(auto candidate_buffer : candidate_buffers) {
      result_types.push_back(candidate_buffer.getType());
    }

    rewriter.setInsertionPoint(compute_op);
    auto results = rewriter.create<lmhlo_disc::EvictOp>(compute_op->getLoc(), result_types, candidate_buffers).getResults();
    
    for(int idx = 0; idx < results.size(); ++idx) {
      // Replace all usage after the trigger point
      candidate_buffers[idx].replaceUsesWithIf(results[idx], [&](OpOperand& use) {
          auto op = use.getOwner();
          int64_t op_int64_ptr = reinterpret_cast<int64_t>(op);
          //if(operation_position_map_.find(op_int64_ptr) == operation_position_map_.end()) return false;
          return operation_position_map_[op_int64_ptr] >
                 trigger_point_position;
        });
    }

    return true;
  }

  SmallVector<Value> CollectOpInputs(Operation* op) {
    SmallVector<Value> inputs;

    if(isa<lmhlo::FusionOp>(op)) {

      std::unordered_set<int64_t> inputs_set;
      op->walk([&](Operation* inner_op) {
        for(auto input : inner_op->getOperands()) {
          if(!NeedSkip(input) && inputs_set.find(reinterpret_cast<int64_t>(input.getAsOpaquePointer())) == inputs_set.end()) {
            inputs.push_back(input);
            inputs_set.insert(reinterpret_cast<int64_t>(input.getAsOpaquePointer()));
          }
        }
      });
    } else {
      for(auto input : op->getOperands()) {
        inputs.push_back(input);
      }
    }

    return inputs;
  }
  bool CheckAndRegenerateInputs(OpBuilder& rewriter, Operation* op, int64_t trigger_point_position) {

    auto all_inputs = CollectOpInputs(op);
    bool changed = false;
    for(auto input : all_inputs) {
      rewriter.setInsertionPoint(op);
      if(input.getDefiningOp() != nullptr && isa<lmhlo_disc::EvictOp>(input.getDefiningOp())) {

        changed = true;
        // This input is a eviction candidate buffer
        // Insert scf & regeneration block
        auto result_type = MemRefType::get({}, rewriter.getI1Type());
        auto check_res = rewriter.create<lmhlo_disc::EvictionCheckOp>(op->getLoc(), result_type, input).getResult(0);
        auto check_res_scalar = rewriter.create<memref::LoadOp>(op->getLoc(), check_res);
        if(check_res_scalar.getType().isa<IntegerType>()) {
          auto res_type = input.getType();
          
          auto if_op = rewriter.create<scf::IfOp>(op->getLoc(), res_type, check_res_scalar, true);
          if_op.getOperation()->setAttr("remat_block", rewriter.getStringAttr("true"));
          
          if_op.getThenRegion().front().clear();
          if_op.getElseRegion().front().clear();

          Block* then_block = &if_op.getThenRegion().getBlocks().front();
          Block* else_block = &if_op.getElseRegion().getBlocks().front();

          auto original_buffer = input;

          /*
          auto evict_op = input.getDefiningOp<lmhlo_disc::EvictOp>();
          for(auto idx = 0; idx < evict_op->getNumResults(); idx++) {
            if(evict_op->getResult(idx) == input) {
              original_buffer = evict_op->getOperand(idx);
              break;
            }
          }
          */

          // Insert recomputation part
          RematEvalResult eval_res{ original_buffer, input, (double)kDefaultValidScore, RematStrategy::kRecompute, {}}; 
          SearchAndCreateRecomputationSubgraph(rewriter, eval_res.original_buffer, eval_res, trigger_point_position);
          InsertRecomputationSubgraph(/*op, rewriter, */then_block, eval_res);

          auto then_terminator_op = rewriter.create<scf::YieldOp>(op->getLoc(), /*operands=*/eval_res.recomputed_buffer);
          then_terminator_op.getOperation()->moveBefore(then_block, then_block->end());
          
          // Insert Else part
          auto else_terminator_op = rewriter.create<scf::YieldOp>(op->getLoc(), /*operands=*/input);
          else_terminator_op.getOperation()->moveBefore(else_block, else_block->end());

          // Replace usage
          input.replaceUsesWithIf(if_op.getResult(0), [&](OpOperand& use) {
            auto op = use.getOwner();
            int64_t op_int64_ptr = reinterpret_cast<int64_t>(op);
            if(operation_position_map_.find(op_int64_ptr) == operation_position_map_.end()) return false;
            return operation_position_map_[op_int64_ptr] >=
                  trigger_point_position;
          });
        }
      }
    }

    return changed;
  }

  bool InsertEvictionAndRegenerationOps(OpBuilder& rewriter, Operation* computation_op, int trigger_point_position) {

    // Already processed, skip
    bool changed = false;
    if(computation_op->hasAttr("already_recomputed")) return changed;

    /******1. Filter some buffers from all living buffers to reduce cpu check overhead********/

    if(!computation_op->hasAttr("already_evicted")) {
      std::vector<RematEvalResult> candidate_evals;
      for (auto item : history_items_.GetHistoryItems()) {
        // Filter out dead buffers
        if (item.live_range.back() < trigger_point_position ||
            item.live_range.front() >= trigger_point_position || item.is_persistent ||
            item.live_out || item.info_outdated) {
          continue;
        }

        // Filter a living buffer out If:
        // 1. It is too small so we get merely no benefit recomputing it.
        // 2. It will be used imediately after this trigger_point_position.
        // 3. Recompute this buffer's defining op will not save any memory or even cause higher memory consumption 
        // since the input for this op, which could be already released, need to be living across this trigger point 
        // 4. The defining op is a collective op
        auto eval_res = EvalRecomputation(rewriter, trigger_point_position, item);
        if (eval_res.score != kInvalidScore) {
          candidate_evals.push_back(eval_res);
        }
      }
      /******2. Insert Evict Op********/
      changed = changed || EvictLivingCandidates(rewriter, candidate_evals, computation_op, trigger_point_position);

      computation_op->setAttr("already_evicted", rewriter.getStringAttr("true"));

      if(changed) return changed;
    }
  
    /******3. Check and Regenerate all inputs********/
    changed = changed || CheckAndRegenerateInputs(rewriter, computation_op, trigger_point_position);

    computation_op->setAttr("already_recomputed", rewriter.getStringAttr("true"));

    return changed;
  }

  const RematStats& GetRematStats() { return remat_stats_; }

  bool NeedSkip(Value buffer) {
    auto item = history_items_.GetItem(buffer);
    return NeedSkip(item);
  }
  bool NeedSkip(const Item& item) {
    /*
    We also need to handle those buffers which are used only inside a fusion
    block We escape them because they will be removed later The pattern is like:
      buffer1 = alloc()
      buffer2 = alloc()
      fusion {
      op(buffer1, buffer2)
      use(buffer2, buffer3)
      }
      dealloc(buffer1)
      dealloc(buffer2)
    Then we know buffer2 can be removed
    */
    if (item.is_persistent || item.live_out ||
        item.compacted_live_range.size() != 3)
      return false;

    if (isa<memref::AllocOp>(reinterpret_cast<Operation*>(
            reverse_operation_position_map_[item.compacted_live_range[0]])) &&
        isa<memref::DeallocOp>(reinterpret_cast<Operation*>(
            reverse_operation_position_map_[item.compacted_live_range[2]]))) {
      return true;
    }

    return false;
  }

  void ResetStatus() {
    operation_position_map_.clear();
    reverse_operation_position_map_.clear();
    history_items_.ResetStatus();
  }

 private:
  HistoryItems history_items_;

  const size_t kSmallMemrefSize =
      10ll *
      1024ll;  // memrefs under kSmallMemrefSize are not considered when remat;
  const int kMaxTryCount = 500;
  const double kInvalidScore = -1;
  const double kDefaultValidScore = 1;
  const int kMinimumInterval = 20;
  const double kMinimumMemorySave =
      1.0 * 1024.0;  // minimum memory saving we want to get

  const int kMaxRecomputeBlockSize = 10;
  std::unordered_map<int64_t, int> operation_position_map_;
  std::unordered_map<int, int64_t> reverse_operation_position_map_;
  RematStrategy remat_strategy_ = RematStrategy::kRecompute;
  RematStats remat_stats_{0, 0, 0};  
};

struct DiscDynamicEvictPass
    : public DiscDynamicEvictPassBase<DiscDynamicEvictPass> {
  using DiscDynamicEvictPassBase<
      DiscDynamicEvictPass>::DiscDynamicEvictPassBase;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<lmhlo_disc::LmhloDiscDialect, memref::MemRefDialect>();
  }

 private:
  MemoryUsageTracker memory_usage_tracker_;
  const int kMaxTryCount = 500;

 public:
  DiscDynamicEvictPass() {}

  void runOnOperation() override {
    auto& context = getContext();
    RewritePatternSet patterns(&context);
    ConversionTarget target(context);
    target.addLegalDialect<arith::ArithDialect, lmhlo_disc::LmhloDiscDialect,
                           memref::MemRefDialect, shape::ShapeDialect,
                           tensor::TensorDialect>();

    ModuleOp module = getOperation();


    // If shape_constraint_graph is not found, we assume it is a static shape graph 
    // We dont process static shape here
    if(!module.lookupSymbol<mlir::func::FuncOp>("shape_constraint_graph")) {
      return;
    }

    auto main_func = module.lookupSymbol<mlir::func::FuncOp>("main");
    OpBuilder builder(main_func.getBody());

    OpBuilder rewriter(&context);

    std::unordered_map<int64_t, int> op_position_map;
    std::unordered_map<int, int64_t> reverse_op_position_map;

    int try_count = 0;
    while (try_count++ < kMaxTryCount) {
      memory_usage_tracker_.ResetStatus();
      op_position_map.clear();
      reverse_op_position_map.clear();
      int op_position = 0;
      int total_op_count = 0;
      for (auto& block : main_func.getBody()) {
        for (auto& op : block) {
          op_position_map[reinterpret_cast<int64_t>(&op)] = op_position;
          reverse_op_position_map[op_position] = reinterpret_cast<int64_t>(&op);
          op_position += 1;
          if (llvm::isa<lmhlo::FusionOp>(op)) {
            op.walk([&](Operation* inner_op) {
              op_position_map[reinterpret_cast<int64_t>(inner_op)] =
                  op_position;
              reverse_op_position_map[op_position] =
                  reinterpret_cast<int64_t>(inner_op);
              op_position += 1;
            });
          }
        }
      }

      total_op_count = op_position;

      memory_usage_tracker_.SetAllOperationPositionInfo(
          op_position_map, reverse_op_position_map);

      // Process all arguments
      for (auto& param : main_func.getArguments()) {
        memory_usage_tracker_.ProcessArgument(param);
      }

      // iterate over op_position_map
      op_position = 0;
      for (auto& block : main_func.getBody()) {
        for (auto& op : block) {
          if (isa<memref::DeallocOp>(op)) {
            // Release a buffer
            memory_usage_tracker_.ProcessDealloc(cast<memref::DeallocOp>(op));
            op_position += 1;
          } else if (isa<lmhlo_disc::CustomCallV2Op, memref::AllocOp,
                         lmhlo::ConstantOp, lmhlo_disc::EvictOp, lmhlo_disc::EvictionCheckOp, scf::IfOp>(op)) {
            // Create a buffer
            if (isa<memref::AllocOp>(op)) {
              memory_usage_tracker_.ProcessAlloc(cast<memref::AllocOp>(op));
            } else if (isa<lmhlo_disc::CustomCallV2Op>(op)) {
              // CustomCallV2Op is also a computation op
              bool changed = memory_usage_tracker_.InsertEvictionAndRegenerationOps(rewriter, &op, op_position);
              if(changed) break;
              
              memory_usage_tracker_.ProcessCustomCallV2(
                  cast<lmhlo_disc::CustomCallV2Op>(op));
            } else if (isa<lmhlo::ConstantOp>(op)) {
              memory_usage_tracker_.ProcessConstant(
                  cast<lmhlo::ConstantOp>(op));
            } else if (isa<lmhlo_disc::EvictOp>(op)) {
              memory_usage_tracker_.ProcessRematOp(
                  cast<lmhlo_disc::EvictOp>(op));
            } else if (isa<lmhlo_disc::EvictionCheckOp>(op)) {
              memory_usage_tracker_.ProcessRematOp(
                  cast<lmhlo_disc::EvictionCheckOp>(op));
            } else if (isa<scf::IfOp>(op)) {
              memory_usage_tracker_.ProcessScfIfOp(
                  cast<scf::IfOp>(op));
            }
            op_position += 1;
          } else if(op.getName().getDialectNamespace() == "arith" 
                   || op.getName().getDialectNamespace() == "memref") {
            op_position += 1;
          } else {
            bool changed = memory_usage_tracker_.InsertEvictionAndRegenerationOps(rewriter, &op, op_position);
            if(changed) break;
            // Update op_position
            op_position += 1;
            if (llvm::isa<lmhlo::FusionOp>(op)) {
              op.walk([&](Operation* inner_op) { op_position += 1; });
            }
          }
        }
      }

      // Means we have already reached the end of the graph
      if (op_position >= total_op_count) {
        break;
      }
    }
    return;
  }
};
}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createDiscDynamicEvictPass() {
  return std::make_unique<DiscDynamicEvictPass>();
}

}  // namespace mhlo_disc
}  // namespace mlir

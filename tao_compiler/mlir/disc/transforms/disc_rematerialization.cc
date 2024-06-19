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
  // peak memory until this item
  size_t current_memory_usage;
};

class LivingItems {
 public:
  LivingItems() = default;

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
                false,
                -1};
  }

  void Add(const Value memref,
           std::unordered_map<int64_t, int>& op_position_map) {
    // Add memrefs and their live range.
    auto item = this->ConstructItemFromValue(memref, op_position_map);
    Add(item);
  }

  void Add(const Item& item) {
    int64_t key = reinterpret_cast<int64_t>(item.memref.getAsOpaquePointer());
    live_range_map_[key] = living_items_.size();
    living_items_.push_back(item);
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
    living_items_[index] = target_item;
  }

  Item& GetItem(Value memref) {
    auto index =
        live_range_map_[reinterpret_cast<int64_t>(memref.getAsOpaquePointer())];
    return living_items_[index];
  }

  std::vector<Item>& GetLivingItems() { return living_items_; }

  void ResetStatus() {
    living_items_.clear();
    live_range_map_.clear();
  }

 private:
  std::vector<Item> living_items_;
  std::map<int64_t, int> live_range_map_;
};

class MemoryUsageTracker {
 public:
  struct RematEvalResult {
    Item item;
    double score;
    size_t memory_saving;
    RematStrategy strategy;
    std::vector<Operation*> remat_block;
    Operation* insertion_point;
    std::string reason;
  };

  struct RematStats {
    int recomputed_item_count;
    int offloaded_item_count;
    int compressed_item_count;
  };

  MemoryUsageTracker() = default;

  void SetRematStrategy(RematStrategy strategy) { remat_strategy_ = strategy; }

  void SetAllOperationPositionInfo(
      const std::unordered_map<int64_t, int>& operation_position_map,
      const std::unordered_map<int, int64_t>& reverse_operation_position_map) {
    operation_position_map_ = operation_position_map;
    reverse_operation_position_map_ = reverse_operation_position_map;
  }

  void ProcessAlloc(memref::AllocOp op) {
    auto memref = op.getResult();
    auto item =
        living_items_.ConstructItemFromValue(memref, operation_position_map_);
    if (NeedSkip(item)) {
      return;
    }
    if (!item.inplace_reuse) {
      current_memory_usage_ += GetMemoryUsageForValue(memref);
      current_peak_memory_ = (current_memory_usage_ > current_peak_memory_)
                                 ? current_memory_usage_
                                 : current_peak_memory_;
    }
    item.current_memory_usage = current_memory_usage_;
    living_items_.Add(item);
  }

  void ProcessDealloc(memref::DeallocOp op) {
    auto memref = op.getOperand();
    if (!living_items_.IsExist(memref)) {
      return;
    }
    current_memory_usage_ -= GetMemoryUsageForValue(memref);
  }

  void ProcessCustomCallV2(lmhlo_disc::CustomCallV2Op op) {
    for (auto memref : op.getResults()) {
      auto item =
          living_items_.ConstructItemFromValue(memref, operation_position_map_);
      if (!item.inplace_reuse) {
        current_memory_usage_ += GetMemoryUsageForValue(memref);
        current_peak_memory_ = (current_memory_usage_ > current_peak_memory_)
                                   ? current_memory_usage_
                                   : current_peak_memory_;
      }
      living_items_.Add(item);
    }
  }

  void ProcessConstant(lmhlo::ConstantOp op) {
    auto memref = op.getOperand();
    auto item =
        living_items_.ConstructItemFromValue(memref, operation_position_map_);
    item.is_persistent = true;
    living_items_.Add(item);
    current_memory_usage_ += GetMemoryUsageForValue(memref);
    current_peak_memory_ = (current_memory_usage_ > current_peak_memory_)
                               ? current_memory_usage_
                               : current_peak_memory_;
  }

  void ProcessArgument(Value memref) {
    auto item =
        living_items_.ConstructItemFromValue(memref, operation_position_map_);
    item.is_persistent = true;
    living_items_.Add(item);
    current_memory_usage_ += GetMemoryUsageForValue(memref);
    current_peak_memory_ = (current_memory_usage_ > current_peak_memory_)
                               ? current_memory_usage_
                               : current_peak_memory_;
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
      throw std::logic_error(
          "GetMemoryUsageForValue for dynamic shape memref not implemented");
    }
  }

  CompactShape GetCompactShape(Value memref) {
    throw std::logic_error("GetCompactShape not implemented");
  }
  size_t GetCompactedMemoryUsageForItem(const Item& item) {
    throw std::logic_error("GetCompactedMemoryUsageForItem not implemented");
  }

  void Recompute(OpBuilder& builder, int op_position, RematEvalResult& result) {
    /*******ONLY Support Single Op With Single Output Recomputation
     * Now*********/
    assert(result.remat_block.size() == 1);

    /*******0. Insert recomputation subgraph of result.item.memref*********/
    builder.setInsertionPoint(result.insertion_point);
    auto new_result_memref = builder.create<memref::AllocOp>(
        result.insertion_point->getLoc(),
        result.item.memref.getType().cast<MemRefType>());

    SmallVector<Value> operands(result.remat_block[0]->getOperands());
    operands.pop_back();
    operands.push_back(new_result_memref);

    OperationState state(result.remat_block[0]->getLoc(),
                         result.remat_block[0]->getName());
    state.addOperands(operands);
    state.addAttributes(result.remat_block[0]->getAttrs());

    auto recompute_op = builder.create(state);
    result.item.memref.replaceUsesWithIf(
        new_result_memref, [&](OpOperand& use) {
          auto op = use.getOwner();
          return operation_position_map_[reinterpret_cast<int64_t>(op)] >=
                 op_position;
        });

    while (true) {
      if (result.item.live_range.back() < op_position) break;
      result.item.live_range.pop_back();
    }

    /*******1. Extend Input Operands's Lifetime*********/
    for (int i = 0; i < operands.size() - 1; i++) {
      if (!living_items_.LiveAcross(
              operands[i], operation_position_map_[reinterpret_cast<int64_t>(
                               result.insertion_point)])) {
        auto& item = living_items_.GetItem(operands[i]);
        item.info_outdated = true;
        living_items_.Update(item.memref, item);
        auto dealloc_op = reinterpret_cast<Operation*>(
            reverse_operation_position_map_[item.live_range.back()]);
        assert(isa<memref::DeallocOp>(dealloc_op));
        dealloc_op->moveAfter(result.insertion_point);
      }
    }

    /*******2. Dealloc Original Memref After Last Use*********/
    auto last_use_op = reinterpret_cast<Operation*>(
        reverse_operation_position_map_[result.item.live_range.back()]);
    if (auto parent_op = last_use_op->getParentOfType<lmhlo::FusionOp>()) {
      last_use_op = parent_op;
    }
    builder.setInsertionPointAfter(last_use_op);
    builder.create<memref::DeallocOp>(last_use_op->getLoc(),
                                      result.item.memref);

    /*******3. Update Status*********/
    current_memory_usage_ -= result.memory_saving;
    living_items_.Update(result.item.memref, result.item);
    remat_stats_.recomputed_item_count++;
  }

  void Offload(OpBuilder& builder, int op_position, RematEvalResult& result) {
    throw std::logic_error("Offload not implemented");
  }

  void Compress(OpBuilder& builder, int op_position, RematEvalResult& result) {
    throw std::logic_error("Compress not implemented");
  }

  RematEvalResult EvalOffload(int op_position, const Item& item,
                              size_t target_peak_memory = -1) {
    throw std::logic_error("EvalOffload not implemented");
  }

  RematEvalResult EvalCompression(int op_position, const Item& item,
                                  size_t target_peak_memory = -1) {
    throw std::logic_error(" EvalCompression not implemented");
  }

  RematEvalResult EvalRecomputationLocal(int op_position, const Item& item,
                                         size_t target_peak_memory = -1) {
    // Memref already released before op_position
    RematEvalResult eval_res{
        item, (double)kInvalidScore, 0, RematStrategy::kRecompute, {}, nullptr,
        ""};

    // We want memory_saving to be large
    if (GetMemoryUsageForValue(item.memref) < kSmallMemrefSize) {
      return eval_res;
    }

    // In SSA format, defining_op is the op that write valid data into memref
    Operation* defining_op = nullptr;
    auto is_write_op = [&](Operation* op, Value memref) {
      if (isa<lmhlo_disc::CustomCallV2Op>(op)) {
        for (auto result : op->getResults()) {
          if (memref == result) return true;
        }
        return false;
      }
      int num_input_operand =
          op->getNumOperands() - disc_ral::getNumResultOperands(op);
      for (auto idx = num_input_operand; idx < op->getNumOperands(); ++idx) {
        if (memref == op->getOperand(idx)) {
          return true;
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

    auto only_one_output = [&](Operation* op) {
      if (isa<lmhlo_disc::CustomCallV2Op>(op)) {
        return op->getNumResults() == 1;
      }
      return disc_ral::getNumResultOperands(op) == 1;
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
        // across different ranks
        if (is_comm_op(user)) {
          return eval_res;
        }

        // We only consider the case where the defining op has only one output
        if (!only_one_output(user)) {
          return eval_res;
        }
        defining_op = user;
      }

      if (item.live_range[idx] > op_position) {
        // We dont recompute operation that create this memref
        if (idx == 1) {
          return eval_res;
        }

        // We can just move dealloc here, dont need to recommpute
        if (idx == item.live_range.size() - 1 &&
            isa<memref::DeallocOp>(reinterpret_cast<Operation*>(
                reverse_operation_position_map_[item.live_range[idx]]))) {
          return eval_res;
        }
        end_position = item.live_range[idx];
        start_position = item.live_range[idx - 1];
        break;
      }
    }

    // We dont want the interval between current position and next time we use
    // this memref to be too small When we count the interval, we exclude
    // alloc/dealloc/constant/argsmutation and ops inside the same fusion block
    Operation* last_fusion_parent = nullptr;
    int temp_position = op_position;
    int interval = end_position - op_position;
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
      if (isa<lmhlo_disc::ArgsMutationOp, lmhlo::ConstantOp, memref::AllocOp,
              memref::AllocaOp, memref::DeallocOp>(reinterpret_cast<Operation*>(
              reverse_operation_position_map_[temp_position]))) {
        interval -= 1;
      }
      temp_position += 1;
    }
    if (interval < kMinimumInterval) {
      return eval_res;
    }

    // We only consider recomputation graph size = 1
    // Recompute a operation might extend it's inputs's lifetime, which might
    // even increase some memory if we remat this op
    size_t memory_usage = GetMemoryUsageForValue(item.memref);
    size_t expanded_memory_size = 0;
    if (isa<lmhlo_disc::CustomCallV2Op>(defining_op)) {
      for (auto input : defining_op->getOperands()) {
        auto& item = living_items_.GetItem(input);
        if (item.info_outdated) return eval_res;
        if (!living_items_.LiveAcross(item, op_position)) {
          expanded_memory_size += GetMemoryUsageForValue(item.memref);
        }
      }
    } else {
      int num_input_operand = defining_op->getNumOperands() -
                              disc_ral::getNumResultOperands(defining_op);
      for (auto idx = 0; idx < num_input_operand; ++idx) {
        auto& item = living_items_.GetItem(defining_op->getOperand(idx));
        if (item.info_outdated) return eval_res;
        if (!living_items_.LiveAcross(item, op_position)) {
          expanded_memory_size += GetMemoryUsageForValue(item.memref);
        }
      }
    }

    // We dont recompute since no memory saving
    if (memory_usage <= expanded_memory_size) {
      return eval_res;
    }

    double memory_saving = (double)memory_usage - (double)expanded_memory_size;

    // We want to make sure the saving is large enough
    if (memory_saving < kMinimumMemorySave) {
      return eval_res;
    }

    eval_res.score = interval * (memory_saving / 1024);
    eval_res.memory_saving = int64_t(memory_saving);
    eval_res.reason =
        "Output buffer size: " + std::to_string(memory_usage / 1024) +
        "\tInput Total size " + std::to_string(expanded_memory_size / 1024) +
        "\t Saving memory for " + std::to_string(memory_saving / 1024) +
        " With Interval " + std::to_string(interval);
    eval_res.remat_block.push_back(defining_op);
    eval_res.insertion_point = reinterpret_cast<Operation*>(
        reverse_operation_position_map_[end_position]);
    if (auto parent_op =
            eval_res.insertion_point->getParentOfType<lmhlo::FusionOp>()) {
      eval_res.insertion_point = parent_op;
    }

    return eval_res;
  }

  RematEvalResult EvalRecomputation(int op_position, const Item& item,
                                    size_t target_peak_memory = -1) {
    return EvalRecomputationLocal(op_position, item, target_peak_memory);
  }
  RematEvalResult GetRematEvaluation(int op_position, const Item& item) {
    switch (remat_strategy_) {
      case RematStrategy::kRecompute:
        return EvalRecomputation(op_position, item);
      case RematStrategy::kCompress:
        return EvalCompression(op_position, item);
      case RematStrategy::kHostOffload:
        return EvalOffload(op_position, item);
      case RematStrategy::kRecomputeAndHostOffload: {
        auto recompute_eval = EvalRecomputation(op_position, item);
        auto offload_eval = EvalOffload(op_position, item);
        if (recompute_eval.score >= offload_eval.score) return recompute_eval;
        return offload_eval;
      }
      case RematStrategy::kRecomputeAndCompress: {
        auto recompute_eval = EvalRecomputation(op_position, item);
        auto compress_eval = EvalCompression(op_position, item);
        if (recompute_eval.score >= compress_eval.score) return recompute_eval;
        return compress_eval;
      }
      case RematStrategy::kAll: {
        auto recompute_eval = EvalRecomputation(op_position, item);
        auto compress_eval = EvalCompression(op_position, item);
        auto offload_eval = EvalOffload(op_position, item);
        if (recompute_eval.score >= compress_eval.score &&
            recompute_eval.score >= offload_eval.score)
          return recompute_eval;
        if (compress_eval.score >= recompute_eval.score &&
            compress_eval.score >= offload_eval.score)
          return compress_eval;
        return offload_eval;
      }
      default:
        return EvalRecomputation(op_position, item);
    }
  }

  bool RematerializeToTargetMemoryUsage(OpBuilder& builder, int op_position,
                                        size_t peak_memory_target) {
    int try_count = 0;
    bool changed = false;
    while (current_memory_usage_ > peak_memory_target &&
           try_count++ < kMaxTryCount) {
      std::vector<RematEvalResult> items_eval_res;

      /******1. Evaluate All Items********/
      for (auto item : living_items_.GetLivingItems()) {
        if (item.live_range.back() < op_position ||
            item.live_range.front() >= op_position || item.is_persistent ||
            item.live_out || item.info_outdated) {
          continue;
        }
        auto eval_res = GetRematEvaluation(op_position, item);
        if (eval_res.score != kInvalidScore) {
          items_eval_res.push_back(eval_res);
        }
      }
      /******2. Choose The Best Item********/
      std::sort(items_eval_res.begin(), items_eval_res.end(),
                [](auto& a, auto& b) { return a.score > b.score; });
      if (items_eval_res.size() == 0) {
        break;
      }
      auto best_item = items_eval_res[0];

      /******3. Remat The Best Item********/
      changed = true;
      switch (best_item.strategy) {
        case RematStrategy::kRecompute:
          Recompute(builder, op_position, best_item);
          break;
        case RematStrategy::kCompress:
          Compress(builder, op_position, best_item);
          break;
        case RematStrategy::kHostOffload:
          Offload(builder, op_position, best_item);
          break;
        default:;
      }
    }
    return changed;
  }

  void RematerializeToLowestMemoryUsage() {
    // Iterate until we cannot get more memory-saving benefit
    throw std::logic_error("RematerializeToLowestMemoryUsage not implemented");
  }

  size_t GetCurrentPeakMemoryUsage() { return current_peak_memory_; }
  size_t GetCurrentMemoryUsage() { return current_memory_usage_; }

  const RematStats& GetRematStats() { return remat_stats_; }
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
    current_peak_memory_ = 0;
    current_memory_usage_ = 0;
    operation_position_map_.clear();
    reverse_operation_position_map_.clear();
    living_items_.ResetStatus();
  }

 private:
  LivingItems living_items_;
  size_t current_peak_memory_ = 0;
  size_t current_memory_usage_ = 0;

  const size_t kSmallMemrefSize =
      10ll *
      1024ll;  // memrefs under kSmallMemrefSize are not considered when remat;
  const int kMaxTryCount = 500;
  const double kInvalidScore = -1;
  const int kMinimumInterval = 10;
  const double kMinimumMemorySave =
      1.0 * 1024.0;  // minimum memory saving we want to get

  const int kMaxRecomputeBlockSize = 1;
  std::unordered_map<int64_t, int> operation_position_map_;
  std::unordered_map<int, int64_t> reverse_operation_position_map_;
  RematStrategy remat_strategy_ = RematStrategy::kRecompute;
  RematStats remat_stats_{0, 0, 0};
};

struct DiscRematerializationPass
    : public DiscRematerializationPassBase<DiscRematerializationPass> {
  using DiscRematerializationPassBase<
      DiscRematerializationPass>::DiscRematerializationPassBase;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<lmhlo_disc::LmhloDiscDialect, memref::MemRefDialect>();
  }

 private:
  MemoryUsageTracker memory_usage_tracker_;
  const int kMaxTryCount = 500;
  size_t peak_memory_limit_in_bytes_;

 public:
  DiscRematerializationPass() {
    peak_memory_limit_in_bytes_ =
        disc_ral::getRematerializationPeakMemoryLimitInBytes();
    VLOG(1) << "Peak memory limit for rematerialization is "
            << peak_memory_limit_in_bytes_ / 1024 / 1024 << " MiB";
  }

  bool IsDynmaicShapeGraph() { return false; }

  size_t GetPeakMemoryLimit() {
    if (IsDynmaicShapeGraph()) {
      return -1;
    }
    return peak_memory_limit_in_bytes_;
  }

  void runOnOperation() override {
    auto& context = getContext();
    RewritePatternSet patterns(&context);
    ConversionTarget target(context);
    target.addLegalDialect<arith::ArithDialect, lmhlo_disc::LmhloDiscDialect,
                           memref::MemRefDialect, shape::ShapeDialect,
                           tensor::TensorDialect>();

    ModuleOp module = getOperation();
    auto main_func = module.lookupSymbol<mlir::func::FuncOp>("main");
    OpBuilder builder(main_func.getBody());

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
            memory_usage_tracker_.ProcessDealloc(cast<memref::DeallocOp>(op));
          } else if (isa<lmhlo_disc::CustomCallV2Op, memref::AllocOp,
                         lmhlo::ConstantOp>(op)) {
            if (isa<memref::AllocOp>(op)) {
              memory_usage_tracker_.ProcessAlloc(cast<memref::AllocOp>(op));
            } else if (isa<lmhlo_disc::CustomCallV2Op>(op)) {
              memory_usage_tracker_.ProcessCustomCallV2(
                  cast<lmhlo_disc::CustomCallV2Op>(op));
            } else if (isa<lmhlo::ConstantOp>(op)) {
              memory_usage_tracker_.ProcessConstant(
                  cast<lmhlo::ConstantOp>(op));
            }
            if (!IsDynmaicShapeGraph() &&
                memory_usage_tracker_.GetCurrentMemoryUsage() >
                    GetPeakMemoryLimit()) {
              auto original_peak_memory =
                  memory_usage_tracker_.GetCurrentMemoryUsage();
              bool changed =
                  memory_usage_tracker_.RematerializeToTargetMemoryUsage(
                      builder, op_position, GetPeakMemoryLimit());
              if (changed) {
                VLOG(1)
                    << "Auto-Rematerialization helped reduce peak memory from "
                    << original_peak_memory / 1024 / 1024 << "MB to "
                    << memory_usage_tracker_.GetCurrentMemoryUsage() / 1024 /
                           1024
                    << "MB at Op position " << op_position;
                break;
              }
            }
          }
          op_position += 1;
          if (llvm::isa<lmhlo::FusionOp>(op)) {
            op.walk([&](Operation* inner_op) { op_position += 1; });
          }
        }
      }

      // Means we have already reached the end of the graph
      if (op_position >= total_op_count) {
        VLOG(1) << "Reached the end of the " << total_op_count
                << " ops, with total "
                << memory_usage_tracker_.GetRematStats().recomputed_item_count
                << " operations recomputed, the final graph will have "
                << memory_usage_tracker_.GetCurrentPeakMemoryUsage() / 1024 /
                       1024
                << " MB peak memory usage\n";
        break;
      }
    }

    // Dynamic Shape Graph Processing
    if (IsDynmaicShapeGraph()) {
      memory_usage_tracker_.RematerializeToLowestMemoryUsage();
    }
    return;
  }
};
}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createDiscRematerializationPass() {
  return std::make_unique<DiscRematerializationPass>();
}

}  // namespace mhlo_disc
}  // namespace mlir

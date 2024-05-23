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

namespace mlir {
using placement_utils::kDiscPlaceAssignment;
using placement_utils::kGpu;

namespace mhlo_disc {
namespace {

template <typename T>
using BaseOpConversion = OpConversionPattern<T>;

enum class ResourceUsageType {
  kNoResource,
  kResourceOccupy,
  kResourceRelease,
};

enum class ResourceType {
  kNoResource = 0,
  kAllToAll = 1,
  kAllGather = 2,
  kAllReduce = 3,
  kCollectivePermute = 4,
  kCopy = 5,
  kReduceScatter = 6,
  kSendRecv = 7,
  kSendHost = 8,
  kRecvHost = 9,
  kCollectiveBroadcast = 10,
  kNumResources = 11,
  kTargetDefinedResourcesBound = 10000,
};

constexpr int64_t ResourceTypeToIndex(ResourceType resource_type) {
  return static_cast<int64_t>(resource_type);
}

constexpr int64_t ResourceUsageTypeToIndex(
    ResourceUsageType resource_usage_type) {
  return static_cast<int64_t>(resource_usage_type);
}

using ResourcePair = std::pair<int64_t, ResourceUsageType>;
using ResourcesVector = std::vector<ResourcePair>;
using TimeCost = double;

class Edge;

struct GraphNode {
  explicit GraphNode(Operation* op, int64_t original_position)
      : op(op), original_position(original_position) {}
  // List of predecessor edges.
  std::vector<Edge> predecessors;
  // List of successor edges.
  std::vector<Edge> successors;

  // Op this Graph node represents
  Operation* op;
  // The prosition of this node in the original order.
  int64_t original_position;
  // Estimated time at which this node is gonna be ready to be scheduled.
  // The node should be added to the ready to be scheduled set when ready_time
  // is less or equal to the current time in the schedule.
  TimeCost ready_time = std::numeric_limits<TimeCost>::max();
  // Number of predecessor nodes this nodes depends on that haven't been
  // scheduled yet.
  int32_t indegree = 0;
  // Number of successor nodes this nodes depends on that haven't been
  // scheduled yet.
  int32_t outdegree = 0;
  // Time cost of the execution of the operation of this nodes represent.
  TimeCost cost = 0.0;
  // Depth in latency terms of a node based on Async operation cost on the path.
  TimeCost async_depth = 0.0;
  // Depth in latency terms of node based on operation cost on the path to the
  // entry node.
  TimeCost depth = 0.0;
  // Depth in latency terms of node based on distance to the entry node.
  int64_t graph_depth = 0;
  // AsyncResources used by the node.
  ResourcesVector resources;
  // Force the scheduling of the nodes with attribute set as late as possible.
  bool force_delay = false;
  // Force the scheduling of the nodes with attribute set as early as possible.
  bool force_early = false;
  // Whether this node has been scheduled or not yet.
  bool scheduled = false;
};

struct Edge {
  Edge(GraphNode* target, TimeCost latency)
      : target(target), latency(latency), original_latency(latency) {}
  // Latency between the two nodes connected by this edge. The other end of the
  // edge is the owner of the HloEdge object. This latency can get updated due
  // to various scheduling optimizations.
  TimeCost latency;
  // Original latency is the initial latency value (typically computed by a
  // latency estimator).
  TimeCost original_latency;
  // Target node of this edge.
  GraphNode* target;
};

// Class used estimate latency between instructions and cost of HLOs.
class LatencyEstimator {
 public:
  LatencyEstimator() = default;
  bool IsAsyncPair(GraphNode* from, GraphNode* target) {
    if (!llvm::isa<lmhlo_disc::CustomCallV2Op>(from->op) ||
        !llvm::isa<lmhlo_disc::CustomCallV2Op>(target->op))
      return false;
    return true;
    auto from_attr = from->op->getAttrOfType<DictionaryAttr>("custom_attrs")
                         .get("async_token_key")
                         .dyn_cast_or_null<mlir::StringAttr>();
    auto target_attr = target->op->getAttrOfType<DictionaryAttr>("custom_attrs")
                           .get("async_token_key")
                           .dyn_cast_or_null<mlir::StringAttr>();
    if (!from_attr || !target_attr) return false;
    return from_attr.getValue() == target_attr.getValue();
  }

  // Uses the approximate or cost model function for GetLatencyBetween based on
  // a flag.
  TimeCost GetLatencyBetween(GraphNode* from, GraphNode* target) {
    if (IsAsyncPair(from, target)) {
      return kHighLatency;
    }
    return kLowLatency;
  }

  // Uses the approximate or cost model function for NodeCost based on a flag.
  TimeCost NodeCost(Operation* op) {
    if (llvm::isa<lmhlo::FusionOp, lmhlo::DotGeneralOp>(op)) {
      return kMediumCost;
    }
    return kLowCost;
  }

  ~LatencyEstimator() = default;

 private:
  static constexpr TimeCost kLowCost = 1.0;
  static constexpr TimeCost kMediumCost = 1000.0;
  static constexpr TimeCost kHighCost = 5000.0;
  // These values are empirically derived to obtain an overlap of one output
  // fusion/convolution with 1 async op or 5 loop fusions with an async op.
  static constexpr TimeCost kLowLatency = 1.0;
  static constexpr TimeCost kHighLatency = 5000.0;
};

struct SchedulerConfig {
  int64_t collective_broadcast_overlap_limit = 1;
  int64_t collective_permute_overlap_limit = 1;
  int64_t all_to_all_overlap_limit = 1;
  int64_t all_gather_overlap_limit = 1;
  int64_t all_reduce_overlap_limit = 1;
  int64_t reduce_scatter_overlap_limit = 1;
  int64_t send_recv_overlap_limit = 1;
  int64_t send_recv_host_overlap_limit = 1;
  int64_t copy_overlap_limit = 1;
  uint64_t memory_limit = UINT64_MAX;
  bool schedule_send_recvs = false;
  // Consider send recv as the same resource. Some platforms do not take well
  // overlapping the send/recv ops between themselves.
  bool force_send_recv_to_use_same_resource = false;
  bool use_real_costmodel = false;
  bool aggressive_scheduling_policies = false;
  bool enable_release_start_policy = false;
  bool resource_sharing = false;
  bool resource_serializing = false;
  bool depthbased_memory_pressure_reduction = false;
  int64_t rerun = 0;
};

class AsyncTracker {
 public:
  AsyncTracker(SchedulerConfig config) : config_(config) {}

  void SetConcurrentResourceLimits(
      std::unordered_map<int64_t, int64_t>& max_concurrent_resource) {
    // Set max_concurrent_resource
    max_concurrent_resource[ResourceTypeToIndex(
        ResourceType::kCollectiveBroadcast)] =
        config_.collective_broadcast_overlap_limit;
    max_concurrent_resource[ResourceTypeToIndex(
        ResourceType::kCollectivePermute)] =
        config_.collective_permute_overlap_limit;
    max_concurrent_resource[ResourceTypeToIndex(ResourceType::kCopy)] =
        config_.copy_overlap_limit;
    max_concurrent_resource[ResourceTypeToIndex(ResourceType::kAllToAll)] =
        config_.all_to_all_overlap_limit;
    max_concurrent_resource[ResourceTypeToIndex(ResourceType::kAllGather)] =
        config_.all_gather_overlap_limit;
    max_concurrent_resource[ResourceTypeToIndex(ResourceType::kAllReduce)] =
        config_.all_reduce_overlap_limit;
    max_concurrent_resource[ResourceTypeToIndex(ResourceType::kReduceScatter)] =
        config_.reduce_scatter_overlap_limit;
    max_concurrent_resource[ResourceTypeToIndex(ResourceType::kSendRecv)] =
        config_.send_recv_overlap_limit;
    max_concurrent_resource[ResourceTypeToIndex(ResourceType::kSendHost)] =
        config_.send_recv_host_overlap_limit;
    max_concurrent_resource[ResourceTypeToIndex(ResourceType::kRecvHost)] =
        config_.send_recv_host_overlap_limit;
  }

  std::string GetResourceName(int64_t resource_type) {
    switch (resource_type) {
      case ResourceTypeToIndex(ResourceType::kNoResource):
        return "kNoResource";
      case ResourceTypeToIndex(ResourceType::kAllToAll):
        return "kAllToAll";
      case ResourceTypeToIndex(ResourceType::kAllGather):
        return "kAllGather";
      case ResourceTypeToIndex(ResourceType::kAllReduce):
        return "kAllReduce";
      case ResourceTypeToIndex(ResourceType::kCollectiveBroadcast):
        return "kCollectiveBroadcast";
      case ResourceTypeToIndex(ResourceType::kCollectivePermute):
        return "kCollectivePermute";
      case ResourceTypeToIndex(ResourceType::kCopy):
        return "kCopy";
      case ResourceTypeToIndex(ResourceType::kSendRecv):
        return "kSendRecv";
      case ResourceTypeToIndex(ResourceType::kSendHost):
        return "kSendHost";
      case ResourceTypeToIndex(ResourceType::kRecvHost):
        return "kRecvHost";
      case ResourceTypeToIndex(ResourceType::kReduceScatter):
        return "kReduceScatter";
      default:
        return "Not a valid default resource";
    }
  }

  std::string GetResourceUsageName(int64_t resource_usage_type) {
    switch (resource_usage_type) {
      case ResourceUsageTypeToIndex(ResourceUsageType::kNoResource):
        return "kNoResource";
      case ResourceUsageTypeToIndex(ResourceUsageType::kResourceOccupy):
        return "kResourceOccupy";
      case ResourceUsageTypeToIndex(ResourceUsageType::kResourceRelease):
        return "kResourceRelease";
      default:
        return "Not a valid resource usage type";
    }
  }

  std::string GetResourceUsageName(ResourceUsageType resource_usage_type) {
    return GetResourceUsageName(ResourceUsageTypeToIndex(resource_usage_type));
  }

  ResourceType getResourceTypeForOp(Operation* op) {
    if (llvm::isa<lmhlo_disc::CustomCallV2Op>(op)) {
      std::string call_target_name =
          op->getAttr("call_target_name").cast<StringAttr>().getValue().str();
      if (call_target_name == "ral_all_gather") {
        return ResourceType::kAllGather;
      } else if (call_target_name == "ral_all_reduce") {
        return ResourceType::kAllReduce;
      } else if (call_target_name == "ral_all_to_all") {
        return ResourceType::kAllToAll;
      } else if (call_target_name == "ral_reduce_scatter") {
        return ResourceType::kReduceScatter;
      }
    }
    return ResourceType::kNoResource;
  }

  bool IsSupportedAsyncDone(Operation* op) {
    return op->hasAttr("call_target_name") &&
           op->getAttr("call_target_name").cast<StringAttr>().getValue() ==
               "ral_async_collective_done";
  }

  bool IsSupportedAsyncStart(Operation* op) {
    return op->hasAttr("call_target_name") &&
           (op->getAttr("call_target_name").cast<StringAttr>().getValue() ==
                "ral_all_gather" ||
            op->getAttr("call_target_name").cast<StringAttr>().getValue() ==
                "ral_all_reduce" ||
            op->getAttr("call_target_name").cast<StringAttr>().getValue() ==
                "ral_reduce_scatter" ||
            op->getAttr("call_target_name").cast<StringAttr>().getValue() ==
                "ral_all_to_all" ||
            op->getAttr("call_target_name").cast<StringAttr>().getValue() ==
                "ral_broadcast");
  }

 private:
  SchedulerConfig config_;
};

// Schedule graph that can be used to drive scheduling
// of instructions.
class ScheduleGraph {
 public:
  // Ops in the list passed to the constructor shouldn't be
  // altered/deleted during the existence of the ScheduleGraph.
  // Nullptr is not a valid value for 'post_order_instructions' and
  // 'alias_analysis'.
  explicit ScheduleGraph(std::vector<Operation*>& post_order_instructions,
                         LatencyEstimator* latency_estimator,
                         AsyncTracker* async_tracker) {
    InitilizeGraphTopology(post_order_instructions, latency_estimator,
                           async_tracker);
    InitializeGraphAnalysis(latency_estimator, async_tracker);
  }

  void ConstructGraphNodes(std::vector<Operation*>& post_order_instructions,
                           LatencyEstimator* latency_estimator,
                           AsyncTracker* async_tracker) {
    int64_t current_pos = 0;
    GraphNode* current_node = nullptr;

    // Construct graph nodes, only consider fusionOps and standalone ops which
    // are not allocs or deallocs.
    for (auto& op : post_order_instructions) {
      auto [new_node_it, inserted] =
          nodes_.try_emplace(op, new GraphNode(op, current_pos));
      op_order_map_[op] = current_pos++;
      new_node_it->second->cost = latency_estimator->NodeCost(op);
    }
  }

  void AddEdges(Operation* op, LatencyEstimator* latency_estimator) {
    auto node = nodes_.at(op);
    std::set<Operation*> dedup_output_ops;

    if (llvm::isa<lmhlo::FusionOp>(op)) {
      // Special process for FusionOp
      op->walk([&](Operation* inner_op) {
        int num_input_operand = inner_op->getNumOperands() -
                                disc_ral::getNumResultOperands(inner_op);
        for (auto idx = num_input_operand; idx < inner_op->getNumOperands();
             ++idx) {
          for (auto* user : inner_op->getOperand(idx).getUsers()) {
            GraphNode* user_node = nullptr;
            if (auto parent_op = user->getParentOfType<lmhlo::FusionOp>()) {
              user = parent_op;
            }
            if (user == op || nodes_.find(user) == nodes_.end() ||
                dedup_output_ops.find(user) != dedup_output_ops.end()) {
              continue;
            }

            dedup_output_ops.insert(user);
            user_node = nodes_.at(user);

            node->successors.push_back(
                Edge(user_node,
                     latency_estimator->GetLatencyBetween(node, user_node)));
            user_node->predecessors.emplace_back(Edge(
                node, latency_estimator->GetLatencyBetween(node, user_node)));
          }
        }
      });
    } else if (llvm::isa<lmhlo_disc::ArgsMutationOp>(op)) {
      for (auto* user : op->getOperand(0).getUsers()) {
        GraphNode* user_node = nullptr;
        if (auto parent_op = user->getParentOfType<lmhlo::FusionOp>()) {
          user = parent_op;
        }
        if (user == op || nodes_.find(user) == nodes_.end() ||
            dedup_output_ops.find(user) != dedup_output_ops.end()) {
          continue;
        }
        user_node = nodes_.at(user);
        if (user_node->original_position < node->original_position) {
          continue;
        }

        dedup_output_ops.insert(user);
        node->successors.push_back(Edge(
            user_node, latency_estimator->GetLatencyBetween(node, user_node)));
        user_node->predecessors.emplace_back(
            Edge(node, latency_estimator->GetLatencyBetween(node, user_node)));
      }
    } else if (disc_ral::isInplaceOperator(op)) {
      Value resultMemref = cast<lmhlo::LmhloOp>(op).getResultBuffer();
      for (auto* user : resultMemref.getUsers()) {
        GraphNode* user_node = nullptr;
        if (auto parent_op = user->getParentOfType<lmhlo::FusionOp>()) {
          user = parent_op;
        }
        if (user == op || nodes_.find(user) == nodes_.end() ||
            dedup_output_ops.find(user) != dedup_output_ops.end()) {
          continue;
        }

        user_node = nodes_.at(user);

        if (user_node->original_position < node->original_position) continue;

        dedup_output_ops.insert(user);
        node->successors.push_back(Edge(
            user_node, latency_estimator->GetLatencyBetween(node, user_node)));
        user_node->predecessors.emplace_back(
            Edge(node, latency_estimator->GetLatencyBetween(node, user_node)));
      }
    } else {
      // Case 1
      for (auto user : op->getUsers()) {
        GraphNode* user_node = nullptr;
        if (auto parent_op = user->getParentOfType<lmhlo::FusionOp>()) {
          user = parent_op;
        }
        if (user == op || nodes_.find(user) == nodes_.end() ||
            dedup_output_ops.find(user) != dedup_output_ops.end()) {
          continue;
        }

        dedup_output_ops.insert(user);
        user_node = nodes_.at(user);

        node->successors.emplace_back(Edge(
            user_node, latency_estimator->GetLatencyBetween(node, user_node)));
        user_node->predecessors.emplace_back(
            Edge(node, latency_estimator->GetLatencyBetween(node, user_node)));
      }

      // Case 2
      int num_input_operand =
          op->getNumOperands() - disc_ral::getNumResultOperands(op);
      for (auto idx = num_input_operand; idx < op->getNumOperands(); ++idx) {
        for (auto* user : op->getOperand(idx).getUsers()) {
          GraphNode* user_node = nullptr;
          if (auto parent_op = user->getParentOfType<lmhlo::FusionOp>()) {
            user = parent_op;
          }
          if (user == op || nodes_.find(user) == nodes_.end() ||
              dedup_output_ops.find(user) != dedup_output_ops.end()) {
            continue;
          }

          dedup_output_ops.insert(user);
          user_node = nodes_.at(user);
          node->successors.push_back(
              Edge(user_node,
                   latency_estimator->GetLatencyBetween(node, user_node)));
          user_node->predecessors.emplace_back(Edge(
              node, latency_estimator->GetLatencyBetween(node, user_node)));
        }
      }
    }
  }

  void InitilizeGraphTopology(std::vector<Operation*>& post_order_instructions,
                              LatencyEstimator* latency_estimator,
                              AsyncTracker* async_tracker) {
    original_order_ = post_order_instructions;

    ConstructGraphNodes(post_order_instructions, latency_estimator,
                        async_tracker);

    // Construct graph topology
    for (auto& op : post_order_instructions) {
      AddEdges(op, latency_estimator);
    }

    for (auto& op : post_order_instructions) {
      auto node = nodes_.at(op);
      node->indegree = node->predecessors.size();
      node->outdegree = node->successors.size();
    }
  }

  void InitializeGraphAnalysis(LatencyEstimator* latency_estimator,
                               AsyncTracker* async_tracker) {
    std::unordered_map<GraphNode*, int> current_rank;
    std::vector<GraphNode*> stack;

    for (auto& op : original_order_) {
      if (auto node = GetNode(op)) {
        current_rank[node] = node->indegree;

        node->async_depth = 0.0;
        node->depth = 0.0;
        node->graph_depth = 0.0;

        if (node->indegree == 0) {
          stack.push_back(node);
        }
      }
    }

    while (!stack.empty()) {
      auto node = stack.back();
      stack.pop_back();
      if (async_tracker->IsSupportedAsyncDone(node->op)) {
        for (auto pred : node->predecessors) {
          node->async_depth = std::max(node->async_depth,
                                       pred.target->async_depth + pred.latency);
          node->depth =
              std::max(node->depth,
                       pred.target->depth + pred.target->cost + pred.latency);
          node->graph_depth =
              std::max(node->graph_depth, pred.target->graph_depth + 1);
          // Set resource for async pair
          if (latency_estimator->IsAsyncPair(pred.target, node)) {
            auto resourceType =
                async_tracker->getResourceTypeForOp(pred.target->op);
            pred.target->resources.push_back(
                std::make_pair<int, ResourceUsageType>(
                    ResourceTypeToIndex(resourceType),
                    ResourceUsageType::kResourceRelease));
            node->resources.push_back(std::make_pair<int, ResourceUsageType>(
                ResourceTypeToIndex(resourceType),
                ResourceUsageType::kResourceOccupy));
          }
        }
      } else {
        for (auto pred : node->predecessors) {
          node->async_depth =
              std::max(node->async_depth, pred.target->async_depth);
          node->depth =
              std::max(node->depth,
                       pred.target->depth + pred.target->cost + pred.latency);
          node->graph_depth =
              std::max(node->graph_depth, pred.target->graph_depth + 1);
        }
      }
      for (auto succ : node->successors) {
        if (--current_rank[succ.target] == 0) {
          stack.push_back(succ.target);
        }
      }
    }
  }

  // Nodes that are close to outputs
  std::vector<GraphNode*> FindBottomRoots() {
    std::vector<GraphNode*> res;
    for (auto& op : original_order_) {
      if (auto node = GetNode(op)) {
        if (node->outdegree == 0) res.push_back(node);
      }
    }
    return res;
  }

  // Nodes that are close to inputs
  std::vector<GraphNode*> FindTopRoots() {
    std::vector<GraphNode*> res;
    for (auto& op : original_order_) {
      if (auto node = GetNode(op)) {
        if (node->indegree == 0) res.push_back(node);
      }
    }
    return res;
  }

  // line of instructions in the original scheduled order. (Before scheduling).
  std::vector<Operation*> GetOriginalOpList() { return original_order_; }

  GraphNode* GetNode(Operation* op) {
    if (nodes_.find(op) == nodes_.end()) {
      return nullptr;
    }
    return nodes_.at(op);
  }

  // Returns what was the original instruction position in the original order.
  int64_t OriginalOpPosition(Operation* op) {
    auto it = op_order_map_.find(op);
    return it->second;
  }

  std::vector<GraphNode*> GetAllNodes() {
    std::vector<GraphNode*> res;
    for (auto iter = nodes_.begin(); iter != nodes_.end(); ++iter) {
      res.push_back(iter->second);
    }
    return res;
  }

 private:
  // Map that allocates the nodes of the graph.
  std::unordered_map<Operation*, GraphNode*> nodes_;
  // Map containing the ordinal value for each instruction.
  std::unordered_map<Operation*, int64_t> op_order_map_;
  // List containing the original order (before scheduling) of the
  // instructions).
  std::vector<Operation*> original_order_;
};

struct SchedulingState {
  using ReadyQueueSet = std::vector<GraphNode*>;
  using ResourceMap = std::unordered_map<int64_t, int64_t>;
  ScheduleGraph sched_graph;
  // Ready set for the nodes. Its ordered by our heuristic defined in
  // ReadySetLt.
  ReadyQueueSet ready_set;
  // Maximum allowed number of overlapping instructions using the key resource
  // type.
  ResourceMap max_concurrent_resource;
  // New scheduling sequence produced by the scheduler. This is in reversed
  // order (because we schedule bottom up). This will be required to be
  // reversed before assigning to the HloSchedule.
  std::vector<Operation*> new_sequence_reversed;
  // Units of time passed in the schedule. To keep track of latency hiding.
  TimeCost current_time = 0;
  // Number of resources in flight.
  ResourceMap resourcesin_flight;
  // Number of instructions using the key resource type in the set waiting to
  // be scheduled.
  ResourceMap resource_users_in_queue;
  // Number of nodes scheduled.
  int64_t scheduled_count = 0;
  // Class returning information about instruction cost and latency between
  // instructions.
  LatencyEstimator* latency_estimator;

  AsyncTracker* async_tracker;

  // Reference to this scheduler run configuration.
  SchedulerConfig config;

  SchedulingState(std::vector<Operation*>& ops,
                  LatencyEstimator* latency_estimator,
                  AsyncTracker* async_tracker, SchedulerConfig& config)
      : sched_graph(ops, latency_estimator, async_tracker),
        async_tracker(async_tracker),
        latency_estimator(latency_estimator),
        config(config) {}
};

class ReadySetLt {
 public:
  explicit ReadySetLt(SchedulingState* sched_state, AsyncTracker* async_tracker)
      : sched_state_(sched_state), async_tracker_(async_tracker) {}

  GraphNode* operator()(GraphNode* a, GraphNode* b) {
    GraphNode* res = nullptr;

    auto willUnlockAsyncDone = [&](GraphNode* node) {
      for (auto pred : node->predecessors) {
        if (async_tracker_->IsSupportedAsyncDone(pred.target->op)) {
          return true;
        }
      }
      return false;
    };

    auto MemoryPressureChange = [&](GraphNode* node) {
      return std::numeric_limits<TimeCost>::max();
    };
    auto isNoOp = [&](GraphNode* node) {
      if (llvm::isa<lmhlo_disc::ArgsMutationOp, lmhlo::TerminatorOp,
                    lmhlo::ConstantOp, func::ReturnOp, memref::AllocOp,
                    memref::AllocaOp>(node->op)) {
        return true;
      }
      return false;
    };

    // 1. Schedule NoOps which have no side-effects
    if (auto ret = chooseBestNode(isNoOp(a), a, isNoOp(b), b)) {
      return ret;
    }

    // 2. Schedule AsyncDone
    if (auto ret =
            chooseBestNode(async_tracker_->IsSupportedAsyncDone(a->op), a,
                           async_tracker_->IsSupportedAsyncDone(b->op), b)) {
      return ret;
    }

    // 3. Schedule nodes that will unlock AsyncDone op
    if (auto ret = chooseBestNode(willUnlockAsyncDone(a), a,
                                  willUnlockAsyncDone(b), b)) {
      return ret;
    }

    TimeCost a_ready_interval = a->ready_time - sched_state_->current_time;
    TimeCost b_ready_interval = b->ready_time - sched_state_->current_time;
    // 4. Schedule ops which cause less stall
    if (auto ret = chooseBestNode(
            a_ready_interval<b_ready_interval, a, a_ready_interval>
                b_ready_interval,
            b)) {
      return ret;
    }

    // 5. Schedule the one which increase least memory pressure
    if (auto ret = chooseBestNode(MemoryPressureChange(a) < 0, a,
                                  MemoryPressureChange(b) < 0, b)) {
      return ret;
    }

    // 6. Fallback strategy, schedule according to the original order
    if (a->original_position < b->original_position) return b;
    return a;
  }

  GraphNode* chooseBestNode(bool condition_a, GraphNode* a, bool condition_b,
                            GraphNode* b) {
    if (condition_a && condition_b) return nullptr;
    if (condition_a) return a;
    if (condition_b) return b;
    return nullptr;
  }

 private:
  SchedulingState* sched_state_;
  AsyncTracker* async_tracker_;
};

class SchedulerCore {
 public:
  SchedulerCore(LatencyEstimator* latency_estimator,
                AsyncTracker* async_tracker, SchedulerConfig schedule_config)
      : latency_estimator_(latency_estimator),
        async_tracker_(async_tracker),
        scheduler_config_(scheduler_config_) {}

  void SchedulingStep(SchedulingState* sched_state) {
    static int step = 1;
    auto node = FindAndExtractBestNodeAvailable(sched_state);
    ScheduleNode(node, sched_state);
    step += 1;
  }

  std::vector<Operation*> ScheduleComputation(
      std::vector<Operation*>& original_op_sequence) {
    SchedulingState sched_state(original_op_sequence, latency_estimator_,
                                async_tracker_, scheduler_config_);

    async_tracker_->SetConcurrentResourceLimits(
        sched_state.max_concurrent_resource);

    auto roots = sched_state.sched_graph.FindBottomRoots();
    for (GraphNode* root : roots) {
      // Set ready time for the roots 0.
      root->ready_time = 0.0;
    }
    sched_state.current_time = 0.0;

    sched_state.ready_set.insert(sched_state.ready_set.end(), roots.begin(),
                                 roots.end());

    while (!sched_state.ready_set.empty()) {
      SchedulingStep(&sched_state);
    }
    // Reverse sched_state.new_sequence_reversed
    // std::reverse(sched_state.new_sequence_reversed.begin(),
    // sched_state.new_sequence_reversed.end());

    return sched_state.new_sequence_reversed;
  }

  GraphNode* FindAndExtractBestNodeAvailable(SchedulingState* sched_state) {
    ReadySetLt ready_lt{sched_state, async_tracker_};
    auto scheduling_instruction_crosses_overlap_limit =
        [&](const GraphNode* node) {
          for (const auto& [resource, limit] :
               sched_state->max_concurrent_resource) {
            // No resources in flight of this kind. Continue.
            auto it = sched_state->resourcesin_flight.find(resource);
            if (it == sched_state->resourcesin_flight.end() ||
                it->second == 0) {
              continue;
            }
            // Number of instances of 'resource' needed if this instruction was
            // to be scheduled.
            int num_resources_needed = 0;
            for (auto& [resource_type_idx, resource_usage_type] :
                 node->resources) {
              if (resource == resource_type_idx) {
                num_resources_needed =
                    resource_usage_type == ResourceUsageType::kResourceOccupy
                        ? 1
                        : 0;
                break;
              }
            }
            if (limit < num_resources_needed) {
              return true;
            }
          }
          return false;
        };

    GraphNode* ready_chosen = nullptr;
    auto chosen_it = sched_state->ready_set.end();

    for (auto ready_node_it = sched_state->ready_set.begin();
         ready_node_it != sched_state->ready_set.end(); ready_node_it++) {
      if (scheduling_instruction_crosses_overlap_limit(*ready_node_it)) {
        continue;
      }

      if (ready_chosen == nullptr) {
        ready_chosen = *ready_node_it;
        chosen_it = ready_node_it;
        continue;
      }

      auto choose_res = ready_lt(ready_chosen, *ready_node_it);
      if (choose_res != ready_chosen) {
        ready_chosen = choose_res;
        chosen_it = ready_node_it;
      }
    }

    sched_state->ready_set.erase(chosen_it);

    return ready_chosen;
  }

  void ScheduleNode(GraphNode* node, SchedulingState* sched_state) {
    // Schedule the node
    sched_state->new_sequence_reversed.push_back(node->op);
    node->scheduled = true;

    // If this node is an async start/done handle the increase/decrease the
    // number of outstanding async ops.
    for (auto& resource : node->resources) {
      if (resource.second == ResourceUsageType::kResourceRelease) {
        ++(sched_state->max_concurrent_resource[resource.first]);
      } else if (resource.second == ResourceUsageType::kResourceOccupy) {
        --(sched_state->max_concurrent_resource[resource.first]);
        --(sched_state->resource_users_in_queue[resource.first]);
      }
    }

    // Compute the new current time after scheduling this node. It is computed
    // as the highest time computed as the sum of the time a successor node has
    // been scheduled and the latency of the edge connecting this node to that
    // node.
    TimeCost schedule_time = sched_state->current_time;
    for (auto& pred : node->successors) {
      TimeCost time_from_edge = pred.target->ready_time + pred.latency;
      schedule_time = std::max(schedule_time, time_from_edge);
    }

    node->ready_time = schedule_time;
    auto current_time = schedule_time + node->cost;

    // Propagate schedule infomation
    int triggered_node_cnt = 0;
    for (auto& pred : node->predecessors) {
      pred.target->outdegree -= 1;
      if (pred.target->outdegree != 0) {
        continue;
      }

      auto ready_time = current_time;
      for (auto& succ : pred.target->successors) {
        TimeCost edge_time = succ.target->ready_time + succ.latency;
        ready_time = std::max(ready_time, edge_time);
      }

      triggered_node_cnt += 1;
      pred.target->ready_time = ready_time;
      sched_state->ready_set.push_back(pred.target);
    }

    for (auto& resource : node->resources) {
      if (resource.second == ResourceUsageType::kResourceRelease) {
        --sched_state->resourcesin_flight[resource.first];
      } else if (resource.second == ResourceUsageType::kResourceOccupy) {
        ++sched_state->resourcesin_flight[resource.first];
      }
    }

    ++sched_state->scheduled_count;
    sched_state->current_time = current_time;
  }

 private:
  LatencyEstimator* latency_estimator_;
  AsyncTracker* async_tracker_;
  SchedulerConfig scheduler_config_;
};

struct DiscOpSchedulePass : public DiscOpSchedulePassBase<DiscOpSchedulePass> {
  using DiscOpSchedulePassBase<DiscOpSchedulePass>::DiscOpSchedulePassBase;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<lmhlo_disc::LmhloDiscDialect, memref::MemRefDialect>();
  }

 private:
  SchedulerConfig scheduler_config_;
  AsyncTracker* async_tracker_;
  LatencyEstimator* latency_estimator_;
  SchedulerCore* scheduler_core_;

 public:
  DiscOpSchedulePass() = default;

  void runOnOperation() override {
    auto& context = getContext();
    RewritePatternSet patterns(&context);
    ConversionTarget target(context);
    target.addLegalDialect<arith::ArithDialect, lmhlo_disc::LmhloDiscDialect,
                           memref::MemRefDialect, shape::ShapeDialect,
                           tensor::TensorDialect>();

    ModuleOp module = getOperation();
    auto main_func = module.lookupSymbol<mlir::func::FuncOp>("main");
    if (!main_func) {
      signalPassFailure();
      return;
    }

    bool need_schedule = false;
    // Initialization
    latency_estimator_ = new LatencyEstimator();
    async_tracker_ = new AsyncTracker(scheduler_config_);

    std::vector<Operation*> original_op_sequence;
    for (auto& block : main_func.getBody()) {
      for (auto& op : block) {
        original_op_sequence.push_back(&op);
        need_schedule =
            need_schedule || async_tracker_->IsSupportedAsyncDone(&op);
      }
    }

    if (!need_schedule) {
      return;
    }

    scheduler_core_ = new SchedulerCore(latency_estimator_, async_tracker_,
                                        scheduler_config_);
    auto scheduled_op_sequence =
        scheduler_core_->ScheduleComputation(original_op_sequence);

    for (auto& block : main_func.getBody()) {
      for (int op_idx = 0; op_idx < scheduled_op_sequence.size(); op_idx++) {
        scheduled_op_sequence[op_idx]->moveBefore(&block.front());
      }
    }

    return;
  }
};
}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createDiscOpSchedulePass() {
  return std::make_unique<DiscOpSchedulePass>();
}

}  // namespace mhlo_disc
}  // namespace mlir

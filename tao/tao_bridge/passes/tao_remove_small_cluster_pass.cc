/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tao_bridge/passes/tao_remove_small_cluster_pass.h"

#include <algorithm>
#include <unordered_set>

#include "absl/strings/str_cat.h"
#include "tao_bridge/tf/const_analysis.h"
#include "tao_bridge/tf/device_util.h"
#include "tao_bridge/tf/xla_cluster_util.h"
#include "tao_bridge/tf/xla_op_registry.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace tao {

Status TaoRemoveSmallClusterPass::CollectOnGraph(Graph *graph) {
  for (Node *n : graph->nodes()) {
    DeviceType device_type("");
    TF_RETURN_IF_ERROR(
        DeviceNameToDeviceType(n->assigned_device_name(), &device_type));
    if (device_type.type_string() != DEVICE_CPU) {
      continue;
    }

    absl::optional<absl::string_view> cluster_str = GetXlaClusterForNode(*n);
    if (!cluster_str.has_value()) {
      continue;
    }

    std::string cluster(*cluster_str);
    if (cluster_nodes_.count(cluster) == 0) {
      std::vector<Node *> tmp_vec;
      cluster_nodes_[cluster] = tmp_vec;
    }

    cluster_nodes_[cluster].push_back(n);

    if (compute_op_.count(n->type_string()) > 0) {
      VLOG(2) << "Node " << n->name() << " with Op " << n->type_string()
              << " added to list.";
      if (compute_op_cnt_.count(cluster) == 0) {
        compute_op_cnt_[cluster] = 1;
      } else {
        compute_op_cnt_[cluster]++;
      }
    }
  }
  VLOG(2) << "Total cluster count is " << cluster_nodes_.size();
  VLOG(2) << "Compute intensive cluster count is " << compute_op_cnt_.size();

  return Status::OK();
}

Status TaoRemoveSmallClusterPass::RemoveNonComputeCluster(Graph *graph) {
  if (cluster_nodes_.empty()) {
    VLOG(2) << "No recorded cluster nodes inofrmation.";
    return Status::OK();
  }
  const int32 kThresholdComCnt = 1;
  std::for_each(cluster_nodes_.begin(), cluster_nodes_.end(),
                [&](std::pair<std::string, std::vector<Node *>> node_map) {
                  if (compute_op_cnt_.count(node_map.first) == 0 ||
                      compute_op_cnt_[node_map.first] < kThresholdComCnt) {
                    VLOG(2) << "Remove cluster " << node_map.first;
                    for (auto n : node_map.second) {
                      VLOG(2) << "Remove node " << n->name() << " "
                              << n->type_string() << " from cluster.";
                      RemoveFromXlaCluster(n);
                    }
                  }
                });

  return Status::OK();
}

Status
TaoRemoveSmallClusterPass::Run(const GraphOptimizationPassOptions &options) {
  // NB!  In this pass we assume the only XLA-auto-clusterable operations that
  // may have side effects are resource variable operations so we don't cluster
  // those.  The pass will have to be updated if this assumption becomes
  // invalid.
  if (!use_tvm_) {
    return Status::OK();
  }

  Graph *graph = options.graph->get();

  TF_RETURN_IF_ERROR(CollectOnGraph(graph));

  TF_RETURN_IF_ERROR(RemoveNonComputeCluster(graph));

  return Status::OK();
}

} // namespace tao
} // namespace tensorflow
